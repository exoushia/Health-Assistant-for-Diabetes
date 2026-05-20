"""
Append-safe JSON writer for enriched_recipes.json — crash-safe incremental export.

Functions:
    default_export_document — empty export skeleton
    load_export / save_export — read/write full document
    get_processed_ids       — recipe ids already in file (resume support)
    append_recipe / append_recipes_batch — atomic append one or many
    _atomic_write           — temp file + rename
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import setup_logger

logger = setup_logger(__name__)

ENRICHED_FILENAME = "enriched_recipes.json"


def default_export_document() -> dict[str, Any]:
    """Empty export document structure."""
    return {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": 0,
        "recipes": [],
    }


def load_export(path: str | Path) -> dict[str, Any]:
    """
    Load enriched recipes JSON or return empty document.

    Args:
        path: Path to enriched_recipes.json.

    Returns:
        Export document with recipes list and metadata.
    """
    path = Path(path)
    if not path.is_file():
        return default_export_document()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "recipes" not in data:
            data = {"version": 1, "recipes": data if isinstance(data, list) else [], "count": 0}
        data.setdefault("recipes", [])
        data["count"] = len(data["recipes"])
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Corrupt export file %s: %s; starting fresh", path, exc)
        return default_export_document()


def get_processed_ids(path: str | Path) -> set[str]:
    """Return set of recipe_id values already in the export file."""
    doc = load_export(path)
    return {r["recipe_id"] for r in doc.get("recipes", []) if r.get("recipe_id")}


def append_recipe(path: str | Path, recipe: dict[str, Any]) -> bool:
    """
    Append a single recipe to the export file (append-safe, atomic write).

    Skips if recipe_id already exists.

    Args:
        path: Output JSON path.
        recipe: Validated recipe dict.

    Returns:
        True if appended, False if duplicate skipped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = load_export(path)
    recipes: list[dict[str, Any]] = doc["recipes"]
    recipe_id = recipe.get("recipe_id")

    if recipe_id and any(r.get("recipe_id") == recipe_id for r in recipes):
        logger.debug("Skipping duplicate recipe_id=%s", recipe_id)
        return False

    recipes.append(recipe)
    doc["recipes"] = recipes
    doc["count"] = len(recipes)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(path, doc)
    return True


def append_recipes_batch(path: str | Path, recipes: list[dict[str, Any]]) -> int:
    """
    Append multiple recipes, skipping duplicates.

    Returns:
        Number of recipes actually appended.
    """
    appended = 0
    for recipe in recipes:
        if append_recipe(path, recipe):
            appended += 1
    return appended


def save_export(path: str | Path, doc: dict[str, Any]) -> Path:
    """Overwrite export file with full document (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc["count"] = len(doc.get("recipes", []))
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(path, doc)
    return path


def _atomic_write(path: Path, doc: dict[str, Any]) -> None:
    """Write JSON via temp file + rename to avoid partial writes."""
    fd, tmp = tempfile.mkstemp(
        suffix=".json",
        dir=path.parent,
        prefix=f".{path.stem}_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
