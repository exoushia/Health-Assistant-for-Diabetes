"""
Offline recipe enrichment: raw CSV → OpenAI nutrition → diabetes_score → JSON export.

Classes:
    RawRecipeRow    — parsed CSV row before enrichment
    EnrichmentStats — counters for a batch run

Functions:
    make_recipe_id / parse_ingredients / parse_steps / parse_csv_row — CSV helpers
    iter_csv_rows           — stream rows from recipes/raw/*.csv
    build_enriched_recipe   — merge row + NutritionEstimate + score
    enrich_single_recipe    — one recipe with retries
    run_enrichment_pipeline — batch job → recipes/processed/enriched_recipes.json
    main                    — CLI entry when run as __main__
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from recipes.schemas.recipe import RecipeSchema
from tools.exporter import ENRICHED_FILENAME, append_recipe, get_processed_ids, load_export
from tools.nutrition import (
    NutritionEstimator,
    classify_meal_type_deterministic,
    classify_veg_deterministic,
    normalize_meal_type,
)
from tools.scoring import apply_diabetes_score_to_recipe
from tools.validation import validate_recipe
from utils.config import PROJECT_ROOT, get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_CSV = PROJECT_ROOT / "recipes" / "raw" / "sample_recipes.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "recipes" / "processed" / ENRICHED_FILENAME


@dataclass
class RawRecipeRow:
    """Parsed row from sample_recipes.csv."""

    name: str
    ingredients: list[str]
    steps: list[str]
    source_url: str
    cuisine: str | None = None
    total_time_mins: int | None = None


@dataclass
class EnrichmentStats:
    """Pipeline run statistics."""

    total_rows: int = 0
    skipped_existing: int = 0
    enriched: int = 0
    failed: int = 0
    validation_failed: int = 0


def make_recipe_id(source_url: str, name: str) -> str:
    """Deterministic recipe_id from URL or name hash."""
    key = source_url.strip() or name.strip()
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"recipe_{digest}"


def parse_ingredients(raw: str) -> list[str]:
    """Split ingredient string into list items."""
    if not raw or not raw.strip():
        return []
    parts = re.split(r",|\n", raw)
    return [p.strip() for p in parts if p.strip()]


def parse_steps(raw: str) -> list[str]:
    """Split instruction blob into step lines."""
    if not raw or not raw.strip():
        return []
    lines = re.split(r"\n+|\.\s+", raw)
    steps = [ln.strip() for ln in lines if len(ln.strip()) > 10]
    return steps[:30] if steps else [raw.strip()[:500]]


def parse_csv_row(row: dict[str, str]) -> RawRecipeRow | None:
    """
    Parse a CSV dict row into RawRecipeRow.

    Returns:
        RawRecipeRow or None if row is unusable.
    """
    name = (row.get("TranslatedRecipeName") or "").strip()
    if not name:
        return None

    cleaned = row.get("Cleaned-Ingredients") or row.get("TranslatedIngredients") or ""
    ingredients = parse_ingredients(cleaned)
    instructions = row.get("TranslatedInstructions") or ""
    steps = parse_steps(instructions)
    url = (row.get("URL") or "").strip()

    time_raw = row.get("TotalTimeInMins") or ""
    total_time: int | None = None
    try:
        total_time = int(float(time_raw)) if time_raw else None
    except ValueError:
        total_time = None

    return RawRecipeRow(
        name=name,
        ingredients=ingredients,
        steps=steps,
        source_url=url,
        cuisine=(row.get("Cuisine") or "").strip() or None,
        total_time_mins=total_time,
    )


def iter_csv_rows(csv_path: Path, *, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Stream rows from the raw recipes CSV."""
    count = 0
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row
            count += 1
            if limit is not None and count >= limit:
                break


def build_enriched_recipe(
    raw: RawRecipeRow,
    estimate: Any,
) -> dict[str, Any]:
    """
    Merge deterministic fields, LLM estimate, and diabetes score into recipe dict.

    Args:
        raw: Parsed CSV row.
        estimate: NutritionEstimate from OpenAI.

    Returns:
        Dict ready for RecipeSchema validation.
    """
    veg = classify_veg_deterministic(raw.ingredients, raw.name)
    meal_type = normalize_meal_type(estimate.meal_type)
    if meal_type == "other":
        meal_type = classify_meal_type_deterministic(raw.name)

    recipe = {
        "recipe_id": make_recipe_id(raw.source_url, raw.name),
        "name": raw.name,
        "meal_type": meal_type,
        "veg": veg,
        "calories": float(estimate.calories),
        "protein": float(estimate.protein),
        "fat": float(estimate.fat),
        "carbs": float(estimate.carbs),
        "fiber": float(estimate.fiber),
        "sugar": float(estimate.sugar),
        "summary": estimate.summary.strip(),
        "ingredients": raw.ingredients,
        "steps": raw.steps,
        "source_url": raw.source_url,
    }
    apply_diabetes_score_to_recipe(recipe)
    return recipe


def enrich_single_recipe(
    raw: RawRecipeRow,
    estimator: NutritionEstimator,
) -> dict[str, Any] | None:
    """
    Enrich one recipe: LLM nutrition + deterministic scoring + validation.

    Returns:
        Validated recipe dict or None on failure.
    """
    estimate = estimator.estimate(
        raw.name,
        raw.ingredients,
        raw.steps,
        cuisine=raw.cuisine,
    )
    if estimate is None:
        return None

    recipe = build_enriched_recipe(raw, estimate)
    result = validate_recipe(recipe)
    if not result["valid"]:
        logger.warning(
            "Validation failed for %s: %s",
            recipe.get("recipe_id"),
            result["errors"],
        )
        return None
    return result["recipe"]


def run_enrichment_pipeline(
    *,
    csv_path: Path | None = None,
    output_path: Path | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> EnrichmentStats:
    """
    Run the full enrichment pipeline.

    Args:
        csv_path: Input CSV path.
        output_path: Output enriched_recipes.json path.
        limit: Max rows to process (None = all).
        resume: Skip recipe_ids already in output file.

    Returns:
        EnrichmentStats summary.
    """
    settings = get_settings()
    csv_path = csv_path or (settings.recipes_raw_dir / "sample_recipes.csv")
    output_path = output_path or (settings.recipes_processed_dir / ENRICHED_FILENAME)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    processed_ids = get_processed_ids(output_path) if resume else set()
    estimator = NutritionEstimator()
    stats = EnrichmentStats()

    rows = list(iter_csv_rows(csv_path, limit=limit))
    stats.total_rows = len(rows)

    logger.info(
        "Starting enrichment: csv=%s output=%s rows=%d resume=%s already=%d",
        csv_path,
        output_path,
        stats.total_rows,
        resume,
        len(processed_ids),
    )

    for row in tqdm(rows, desc="Enriching recipes", unit="recipe"):
        parsed = parse_csv_row(row)
        if parsed is None:
            stats.failed += 1
            continue

        recipe_id = make_recipe_id(parsed.source_url, parsed.name)
        if resume and recipe_id in processed_ids:
            stats.skipped_existing += 1
            continue

        enriched = enrich_single_recipe(parsed, estimator)
        if enriched is None:
            stats.failed += 1
            continue

        if not append_recipe(output_path, enriched):
            stats.skipped_existing += 1
            continue

        processed_ids.add(recipe_id)
        stats.enriched += 1
        logger.debug(
            "Enriched %s | diabetes_score=%.3f | meal_type=%s | veg=%s",
            recipe_id,
            enriched.get("diabetes_score"),
            enriched.get("meal_type"),
            enriched.get("veg"),
        )

    doc = load_export(output_path)
    logger.info(
        "Enrichment complete: enriched=%d failed=%d skipped=%d total_in_file=%d",
        stats.enriched,
        stats.failed,
        stats.skipped_existing,
        doc.get("count", 0),
    )
    return stats


def main() -> None:
    """CLI entry point for the enrichment pipeline."""
    parser = argparse.ArgumentParser(description="Enrich raw recipes with OpenAI nutrition")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output enriched_recipes.json path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max recipes to process (for dev/testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process recipes even if already exported",
    )
    args = parser.parse_args()

    stats = run_enrichment_pipeline(
        csv_path=args.csv,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
    print(
        f"Done: enriched={stats.enriched} failed={stats.failed} "
        f"skipped={stats.skipped_existing} / {stats.total_rows}"
    )


if __name__ == "__main__":
    main()
