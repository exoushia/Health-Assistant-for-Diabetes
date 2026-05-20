"""
Health profile schemas — biomarker floats, risk level, conditions, medications.

Classes:
    BiomarkerValues      — hba1c, glucose, cholesterol, triglycerides, vitamin D
    HealthProfileSchema  — full stored profile with timestamps and flags
"""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

DiabetesRiskLevel = Literal["low", "medium", "high", "unknown"]


class BiomarkerValues(BaseModel):
    """Numeric lab values extracted from OCR text (all optional)."""

    hba1c_percent: float | None = None
    fasting_glucose_mg_dl: float | None = None
    cholesterol_mg_dl: float | None = None
    triglycerides_mg_dl: float | None = None
    vitamin_d_ng_ml: float | None = None

    @field_validator(
        "hba1c_percent",
        "fasting_glucose_mg_dl",
        "cholesterol_mg_dl",
        "triglycerides_mg_dl",
        "vitamin_d_ng_ml",
        mode="before",
    )
    @classmethod
    def coerce_numeric(cls, value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)


class HealthProfileSchema(BaseModel):
    """Validated health profile persisted per user."""

    user_id: str | None = None
    source: Literal["ocr", "manual", "api"] = "ocr"
    raw_text_excerpt: str | None = Field(
        default=None,
        description="First N characters of OCR text for audit",
    )
    biomarkers: BiomarkerValues = Field(default_factory=BiomarkerValues)
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    diabetes_risk: DiabetesRiskLevel = "unknown"
    missing_biomarkers: list[str] = Field(default_factory=list)
    extraction_confidence: dict[str, float] = Field(default_factory=dict)
    flags: dict[str, str] = Field(default_factory=dict)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_user_biomarker_dict(self) -> dict[str, Any]:
        """Flatten for orchestrator state and ranking tools."""
        b = self.biomarkers
        out: dict[str, Any] = {
            "hba1c": b.hba1c_percent,
            "fasting_glucose_mg_dl": b.fasting_glucose_mg_dl,
            "cholesterol_mg_dl": b.cholesterol_mg_dl,
            "triglycerides_mg_dl": b.triglycerides_mg_dl,
            "vitamin_d_ng_ml": b.vitamin_d_ng_ml,
            "conditions": self.conditions,
            "medications": self.medications,
            "diabetes_risk": self.diabetes_risk,
            "missing_biomarkers": self.missing_biomarkers,
        }
        out.update(self.flags)
        return out
