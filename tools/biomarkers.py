"""
Deterministic biomarker parsing from OCR text — regex only, no LLM.

Functions:
    extract_biomarkers          — HbA1c, glucose, lipids, conditions, meds from raw text
    assign_diabetes_risk        — high/medium/low/unknown from HbA1c thresholds
    validate_health_profile     — Pydantic HealthProfileSchema check
    build_health_profile        — extract + risk + validate + optional save
    save_health_profile_for_user / load_health_profile — data/health/{user_id}/
    parse_biomarkers            — accept dict or string input
    biomarker_summary           — one-line human summary for UI
    _first_match / _validate_*    — regex capture with range validation
    _extract_conditions / _extract_medications — keyword lists from report text
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tools.scoring import compute_diabetes_score  # re-export for backward compatibility
from recipes.schemas.health_schema import (
    BiomarkerValues,
    DiabetesRiskLevel,
    HealthProfileSchema,
)
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Expected biomarker keys for missing-value guardrails
BIOMARKER_FIELDS = (
    "hba1c_percent",
    "fasting_glucose_mg_dl",
    "cholesterol_mg_dl",
    "triglycerides_mg_dl",
    "vitamin_d_ng_ml",
)

# ---------------------------------------------------------------------------
# Regex patterns (deterministic OCR parsing)
# ---------------------------------------------------------------------------

_HBA1C_PATTERNS = (
    # Lab table layouts (e.g. Orange Health): result on HPLC line, not reference range
    r"(?:HPLC|Chromatography)[^\n]{0,80}?(\d{1,2}(?:\.\d+)?)\s*%\s*(?:pre-?diabetes|diabetes)",
    r"(\d{1,2}(?:\.\d+)?)\s*%\s*pre-?diabetes",
    r"(?:glycated|glycosylated)\s+(?:haemoglobin|hemoglobin)\s*(?:\([^)]*\))?\s*[^\d]{0,40}(\d{1,2}(?:\.\d+)?)\s*%\s*(?:pre-?diabetes|diabetes)",
    r"(?:hb\s*[-]?\s*a\s*1[cC]|hemoglobin\s+a\s*1[cC]|hba1c|hba1[cC]|hbaic)\s*[:\-=]?\s*(\d{1,2}(?:\.\d+)?)\s*%",
)

_FASTING_GLUCOSE_PATTERNS = (
    # Require explicit fasting/FPG — avoids Mean Blood Glucose (eAG) on HbA1c panels
    r"(?:fasting\s+)(?:blood\s+)?glucose(?:\s*\(fasting\))?\s*[:\-=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl|mg/dL|mg%)?",
    r"(?:fpg|fasting\s+blood\s+sugar)\s*[:\-=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl|mg/dL|mg%)?",
    r"glucose\s*,?\s*fasting\s*[:\-=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl|mg/dL|mg%)?",
)

_CHOLESTEROL_PATTERNS = (
    r"(?:total\s+)?cholesterol\s*[:\-=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl|mg/dL)?",
    r"(?:serum\s+)?cholesterol\s*\(total\)\s*[:\-=]?\s*(\d{2,3}(?:\.\d+)?)",
)

_TRIGLYCERIDES_PATTERNS = (
    r"triglycerides?\s*[:\-=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl|mg/dL)?",
    r"\btg\s*[:\-=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:mg/dl|mg/dL)?",
)

_VITAMIN_D_PATTERNS = (
    r"(?:25\s*[-]?\s*oh\s+)?vitamin\s*d\s*(?:3)?\s*[:\-=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:ng/ml|ng/mL|nmol/l)?",
    r"vit\s*d\s*[:\-=]?\s*(\d{1,2}(?:\.\d+)?)",
)

_CONDITION_KEYWORDS: dict[str, str] = {
    r"\btype\s*1\s*diabetes\b": "type_1_diabetes",
    r"\btype\s*2\s*diabetes\b": "type_2_diabetes",
    r"\btype\s*i\s*diabetes\b": "type_1_diabetes",
    r"\btype\s*ii\s*diabetes\b": "type_2_diabetes",
    r"\bdiabetes\s+mellitus\b": "diabetes_mellitus",
    r"\bdiabetes\b": "diabetes",
    r"\bprediabetes\b": "prediabetes",
    r"\bpre[\s-]?diabetes\b": "prediabetes",
    r"\bhypertension\b": "hypertension",
    r"\bhigh\s+blood\s+pressure\b": "hypertension",
    r"\bhyperlipidemia\b": "hyperlipidemia",
    r"\bobesity\b": "obesity",
    r"\bhypothyroidism\b": "hypothyroidism",
    r"\bhyperthyroidism\b": "hyperthyroidism",
    r"\bmetabolic\s+syndrome\b": "metabolic_syndrome",
    r"\bckd\b": "chronic_kidney_disease",
    r"\bchronic\s+kidney\s+disease\b": "chronic_kidney_disease",
}

_MEDICATION_KEYWORDS: dict[str, str] = {
    r"\bmetformin\b": "metformin",
    r"\binsulin\b": "insulin",
    r"\bglipizide\b": "glipizide",
    r"\bglimepiride\b": "glimepiride",
    r"\bsitagliptin\b": "sitagliptin",
    r"\bempagliflozin\b": "empagliflozin",
    r"\bliraglutide\b": "liraglutide",
    r"\bsemaglutide\b": "semaglutide",
    r"\batorvastatin\b": "atorvastatin",
    r"\brosuvastatin\b": "rosuvastatin",
    r"\blisinopril\b": "lisinopril",
    r"\bamlodipine\b": "amlodipine",
    r"\blevothyroxine\b": "levothyroxine",
    r"\baspirin\b": "aspirin",
    r"\bwarfarin\b": "warfarin",
}

# Reference ranges for derived flags (deterministic, not diagnostic)
_GLUCOSE_FASTING_NORMAL = (70, 99)
_HBA1C_TARGET_PERCENT = 7.0


def extract_biomarkers(raw_text: str) -> dict[str, Any]:
    """
    Extract biomarkers, conditions, and medications from Tesseract OCR text.

    Uses deterministic regex and keyword matching only (no LLM).

    Args:
        raw_text: Cleaned or raw OCR output.

    Returns:
        Dict with biomarker floats (or None), conditions, medications,
        missing_biomarkers, and extraction_confidence per field.
    """
    text = _normalize_text(raw_text)
    if not text.strip():
        logger.warning("extract_biomarkers: empty OCR text")
        return _empty_extraction()

    extracted: dict[str, Any] = {
        "hba1c_percent": _first_match(text, _HBA1C_PATTERNS, _validate_hba1c),
        "fasting_glucose_mg_dl": _first_match(text, _FASTING_GLUCOSE_PATTERNS, _validate_glucose),
        "cholesterol_mg_dl": _first_match(text, _CHOLESTEROL_PATTERNS, _validate_cholesterol),
        "triglycerides_mg_dl": _first_match(text, _TRIGLYCERIDES_PATTERNS, _validate_triglycerides),
        "vitamin_d_ng_ml": _first_match(text, _VITAMIN_D_PATTERNS, _validate_vitamin_d),
        "conditions": _extract_conditions(text),
        "medications": _extract_medications(text),
    }

    missing = [field for field in BIOMARKER_FIELDS if extracted.get(field) is None]
    extracted["missing_biomarkers"] = missing
    extracted["extraction_confidence"] = {
        field: (1.0 if extracted.get(field) is not None else 0.0) for field in BIOMARKER_FIELDS
    }

    logger.info(
        "Extracted biomarkers: found=%d missing=%s conditions=%d meds=%d",
        len(BIOMARKER_FIELDS) - len(missing),
        missing,
        len(extracted["conditions"]),
        len(extracted["medications"]),
    )
    return extracted


def assign_diabetes_risk(data: dict[str, Any]) -> DiabetesRiskLevel:
    """
    Classify diabetes risk from extracted or manual biomarker data.

    Rules (HbA1c %):
        - >= 6.5 → high
        - >= 5.7 → medium
        - < 5.7  → low
        - missing → unknown

    Args:
        data: Dict containing hba1c_percent and/or nested biomarkers key.

    Returns:
        Risk level string.
    """
    hba1c = _resolve_hba1c(data)
    if hba1c is None:
        logger.debug("assign_diabetes_risk: HbA1c missing → unknown")
        return "unknown"
    if hba1c >= 6.5:
        return "high"
    if hba1c >= 5.7:
        return "medium"
    return "low"


def validate_health_profile(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a health profile dict against HealthProfileSchema.

    Args:
        data: Raw or partially structured profile dict.

    Returns:
        Dict with valid (bool), errors (list), and profile (dict | None).
    """
    try:
        profile = HealthProfileSchema.model_validate(data)
        return {
            "valid": True,
            "errors": [],
            "profile": profile.model_dump(),
        }
    except ValidationError as exc:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        logger.warning("Health profile validation failed: %s", errors)
        return {"valid": False, "errors": errors, "profile": None}


def build_health_profile(
    raw_text: str,
    *,
    user_id: str | None = None,
    source: str = "ocr",
    store: bool = True,
) -> dict[str, Any]:
    """
    Full pipeline: OCR text → extract → risk → validate → optional persist.

    Args:
        raw_text: Tesseract OCR output (prefer cleaned text from tools.ocr).
        user_id: When set, save JSON under data/health/{user_id}/.
        source: Provenance label for the profile.
        store: Write profile to disk when user_id is provided.

    Returns:
        Validated health profile as JSON-serializable dict.
    """
    extracted = extract_biomarkers(raw_text)
    risk = assign_diabetes_risk(extracted)

    biomarkers = BiomarkerValues(
        hba1c_percent=extracted.get("hba1c_percent"),
        fasting_glucose_mg_dl=extracted.get("fasting_glucose_mg_dl"),
        cholesterol_mg_dl=extracted.get("cholesterol_mg_dl"),
        triglycerides_mg_dl=extracted.get("triglycerides_mg_dl"),
        vitamin_d_ng_ml=extracted.get("vitamin_d_ng_ml"),
    )

    profile_data = {
        "user_id": user_id,
        "source": source,
        "raw_text_excerpt": raw_text[:500] if raw_text else None,
        "biomarkers": biomarkers.model_dump(),
        "conditions": extracted.get("conditions", []),
        "medications": extracted.get("medications", []),
        "diabetes_risk": risk,
        "missing_biomarkers": extracted.get("missing_biomarkers", []),
        "extraction_confidence": extracted.get("extraction_confidence", {}),
        "flags": _derive_flags(biomarkers),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    validation = validate_health_profile(profile_data)
    if not validation["valid"]:
        logger.error("Built profile failed validation: %s", validation["errors"])
        profile_data["validation_errors"] = validation["errors"]
        return profile_data

    profile = validation["profile"]
    assert profile is not None

    if store and user_id:
        path = save_health_profile_for_user(user_id, profile)
        profile["storage_path"] = str(path)
        logger.info("Health profile stored for %s at %s", user_id, path)

    return profile


def save_health_profile_for_user(user_id: str, profile: dict[str, Any]) -> Path:
    """
    Persist health profile JSON under data/health/{user_id}/.

    Args:
        user_id: Target user identifier.
        profile: Validated profile dict.

    Returns:
        Path to written profile.json.
    """
    settings = get_settings()
    safe_id = re.sub(r"[^\w\-]", "_", user_id.strip()) or "unknown_user"
    user_dir = settings.data_dir / "health" / safe_id
    user_dir.mkdir(parents=True, exist_ok=True)

    out_path = user_dir / "profile.json"
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    pointer = user_dir / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "path": str(out_path),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "diabetes_risk": profile.get("diabetes_risk"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def load_health_profile(user_id: str) -> dict[str, Any] | None:
    """Load the latest stored health profile for a user."""
    settings = get_settings()
    safe_id = re.sub(r"[^\w\-]", "_", user_id.strip()) or "unknown_user"
    path = settings.data_dir / "health" / safe_id / "profile.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load health profile for %s: %s", user_id, exc)
        return None


def parse_biomarkers(raw: dict[str, Any] | str) -> dict[str, Any]:
    """
    Normalize biomarkers for orchestrator state.

    Accepts OCR text (str), health profile dict, or legacy flat biomarker dict.

    Args:
        raw: OCR text, health profile, or biomarker fields.

    Returns:
        Flat dict with values, flags, and diabetes_risk for downstream tools.
    """
    if isinstance(raw, str):
        profile = build_health_profile(raw, store=False)
        return _profile_to_flat(profile)

    if "biomarkers" in raw and isinstance(raw["biomarkers"], dict):
        risk = raw.get("diabetes_risk") or assign_diabetes_risk(raw["biomarkers"])
        merged = {
            **raw["biomarkers"],
            "conditions": raw.get("conditions", []),
            "medications": raw.get("medications", []),
            "diabetes_risk": risk,
            "missing_biomarkers": raw.get("missing_biomarkers", []),
        }
        return _enrich_flags(merged)

    if raw.get("raw_text"):
        return parse_biomarkers(raw["raw_text"])

    return _enrich_flags(dict(raw))


def biomarker_summary(biomarkers: dict[str, Any]) -> str:
    """Return a short human-readable summary for prompts and UI."""
    risk = biomarkers.get("diabetes_risk", "unknown")
    missing = biomarkers.get("missing_biomarkers", [])
    hba1c = biomarkers.get("hba1c") or biomarkers.get("hba1c_percent")
    glucose = biomarkers.get("fasting_glucose_mg_dl")
    parts = [f"diabetes_risk={risk}"]
    if hba1c is not None:
        parts.append(f"HbA1c={hba1c}%")
    if glucose is not None:
        parts.append(f"fasting_glucose={glucose} mg/dL")
    if missing:
        parts.append(f"missing={','.join(missing)}")
    return "Biomarkers: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_extraction() -> dict[str, Any]:
    return {
        **{field: None for field in BIOMARKER_FIELDS},
        "conditions": [],
        "medications": [],
        "missing_biomarkers": list(BIOMARKER_FIELDS),
        "extraction_confidence": {field: 0.0 for field in BIOMARKER_FIELDS},
    }


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _first_match(
    text: str,
    patterns: tuple[str, ...],
    validator: Any,
) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                validated = validator(value)
                if validated is not None:
                    return validated
            except (ValueError, IndexError):
                continue
    return None


def _validate_hba1c(value: float) -> float | None:
    return value if 4.0 <= value <= 20.0 else None


def _validate_glucose(value: float) -> float | None:
    return value if 40.0 <= value <= 500.0 else None


def _validate_cholesterol(value: float) -> float | None:
    return value if 50.0 <= value <= 600.0 else None


def _validate_triglycerides(value: float) -> float | None:
    return value if 20.0 <= value <= 2000.0 else None


def _validate_vitamin_d(value: float) -> float | None:
    return value if 3.0 <= value <= 150.0 else None


def _extract_conditions(text: str) -> list[str]:
    found: list[str] = []
    for pattern, label in _CONDITION_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            if label not in found:
                found.append(label)
    found.extend(_extract_section_items(text, r"(?:conditions?|diagnoses|problems)\s*[:\-]"))
    return sorted(set(found))


def _extract_medications(text: str) -> list[str]:
    found: list[str] = []
    for pattern, label in _MEDICATION_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            if label not in found:
                found.append(label)
    found.extend(_extract_section_items(text, r"(?:medications?|meds|current\s+rx)\s*[:\-]"))
    return sorted(set(found))


def _extract_section_items(text: str, header_pattern: str) -> list[str]:
    """Parse comma- or line-separated items after a section header."""
    items: list[str] = []
    match = re.search(
        header_pattern + r"\s*\n?(.+?)(?:\n\n|\n[A-Z][a-z]+[\s:]|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return items
    block = match.group(1)
    for part in re.split(r"[,;\n]", block):
        cleaned = part.strip().lower()
        if 2 <= len(cleaned) <= 80 and not cleaned.isdigit():
            items.append(cleaned.replace(" ", "_"))
    return items


def _resolve_hba1c(data: dict[str, Any]) -> float | None:
    if "hba1c_percent" in data and data["hba1c_percent"] is not None:
        return float(data["hba1c_percent"])
    if "hba1c" in data and data["hba1c"] is not None:
        return float(data["hba1c"])
    biomarkers = data.get("biomarkers")
    if isinstance(biomarkers, dict):
        return _resolve_hba1c(biomarkers)
    return None


def _derive_flags(biomarkers: BiomarkerValues) -> dict[str, str]:
    flags: dict[str, str] = {}
    if biomarkers.fasting_glucose_mg_dl is not None:
        low, high = _GLUCOSE_FASTING_NORMAL
        g = biomarkers.fasting_glucose_mg_dl
        flags["glucose_flag"] = "low" if g < low else "high" if g > high else "normal"
    if biomarkers.hba1c_percent is not None:
        flags["hba1c_flag"] = (
            "above_target" if biomarkers.hba1c_percent > _HBA1C_TARGET_PERCENT else "at_target"
        )
    if biomarkers.cholesterol_mg_dl is not None and biomarkers.cholesterol_mg_dl >= 200:
        flags["cholesterol_flag"] = "elevated"
    if biomarkers.triglycerides_mg_dl is not None and biomarkers.triglycerides_mg_dl >= 150:
        flags["triglycerides_flag"] = "elevated"
    if biomarkers.vitamin_d_ng_ml is not None and biomarkers.vitamin_d_ng_ml < 20:
        flags["vitamin_d_flag"] = "deficient"
    return flags


def _enrich_flags(data: dict[str, Any]) -> dict[str, Any]:
    biomarkers = BiomarkerValues(
        hba1c_percent=data.get("hba1c_percent") or data.get("hba1c"),
        fasting_glucose_mg_dl=data.get("fasting_glucose_mg_dl"),
        cholesterol_mg_dl=data.get("cholesterol_mg_dl"),
        triglycerides_mg_dl=data.get("triglycerides_mg_dl"),
        vitamin_d_ng_ml=data.get("vitamin_d_ng_ml"),
    )
    enriched = dict(data)
    enriched.update(_derive_flags(biomarkers))
    enriched.setdefault("diabetes_risk", assign_diabetes_risk(enriched))
    if "hba1c" not in enriched and biomarkers.hba1c_percent is not None:
        enriched["hba1c"] = biomarkers.hba1c_percent
    return enriched


def _profile_to_flat(profile: dict[str, Any]) -> dict[str, Any]:
    schema = HealthProfileSchema.model_validate(profile)
    flat = schema.to_user_biomarker_dict()
    flat.update(schema.flags)
    return flat
