"""
Localization — Hindi meal plans for WhatsApp (Sarvam first, OpenAI fallback).

Functions:
    localize_recipe / localize_meal_plan     — deep-copy plan with translated strings
    localize_meal_plan_whatsapp              — full WhatsApp body in Hindi
    translate_to_hindi_conversational        — short phrase translation
    format_meal_plan_whatsapp_en             — English WhatsApp formatter
    clean_markdown / emoji_safe_format       — strip markdown, keep emojis safe
    preserve_numeric_tokens / restore_*      — protect numbers during translation
    truncate_whatsapp                        — Twilio length cap
    _translate_via_sarvam / _translate_via_openai — provider backends
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from openai import OpenAI

from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

_LOCALE_DEFAULT = "en"
_WHATSAPP_MAX_CHARS = 1600

# ---------------------------------------------------------------------------
# Localization prompts
# ---------------------------------------------------------------------------

HINDI_LOCALIZATION_PROMPT = """You convert diabetes meal-plan messages into natural conversational Hindi for WhatsApp.

Rules:
- Use everyday spoken Hindi (आप form), not stiff literal translation.
- Keep the SAME section order, line breaks, bullet structure, and meal slots.
- NEVER remove or change numbers, calories (kcal), macros, or recipe IDs.
- KEEP all emojis exactly as they appear (🍽 🌅 ☀️ 🌙 ✅ etc.).
- Do NOT add markdown (no **, ##, ```).
- Max length ~1500 characters; be concise per meal.
- Day names in Hindi (सोमवार…). Meals: नाश्ता, दोपहर का भोजन, रात का खाना.
- Output ONLY the Hindi message text, nothing else.
"""

HINDI_RECIPE_PROMPT = """Translate this recipe line to natural Hindi for WhatsApp. Keep numbers and emojis unchanged. One short line only."""

# ---------------------------------------------------------------------------
# Labels & emoji maps
# ---------------------------------------------------------------------------

MEAL_EMOJI = {
    "breakfast": "🌅",
    "lunch": "☀️",
    "dinner": "🌙",
    "snack": "🍎",
}

WEEKDAY_EN_TO_HI = {
    "monday": "सोमवार",
    "tuesday": "मंगलवार",
    "wednesday": "बुधवार",
    "thursday": "गुरुवार",
    "friday": "शुक्रवार",
    "saturday": "शनिवार",
    "sunday": "रविवार",
}

MEAL_TYPE_EN_TO_HI = {
    "breakfast": "नाश्ता",
    "lunch": "दोपहर का भोजन",
    "dinner": "रात का खाना",
    "snack": "हल्का नाश्ता",
}

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def localize_recipe(recipe: dict[str, Any], locale: str = _LOCALE_DEFAULT) -> dict[str, Any]:
    """
    Return recipe with localized fields.

    For Hindi, adds name_hi and summary_hi when translation is available.
    """
    localized = dict(recipe)
    translations = recipe.get("translations", {})
    if locale in translations:
        localized.update(translations[locale])

    if "title" in localized and "name" not in localized:
        localized["name"] = localized["title"]

    if _is_hindi(locale):
        name = localized.get("name", "")
        summary = localized.get("summary", "")
        if name:
            localized["name_hi"] = _translate_short_text(
                f"{name}. {summary}" if summary else name,
                context="recipe_title",
            )
        if summary:
            localized["summary_hi"] = _translate_short_text(summary, context="recipe_summary")

    localized["locale"] = locale
    return localized


def localize_meal_plan(plan: dict[str, Any], locale: str = _LOCALE_DEFAULT) -> dict[str, Any]:
    """
    Localize meal plan structure and attach WhatsApp-ready text.

    Preserves original English fields; adds localized copies and whatsapp_text.
    """
    result = {**plan, "locale": locale}
    result["whatsapp_text_en"] = format_meal_plan_whatsapp_en(plan)

    if _is_hindi(locale):
        result["whatsapp_text"] = localize_meal_plan_whatsapp(plan, locale=locale)
        result["days_localized"] = _localize_plan_days_structure(plan, locale)
    else:
        result["whatsapp_text"] = result["whatsapp_text_en"]

    return result


def localize_meal_plan_whatsapp(
    plan: dict[str, Any],
    locale: str = "hi",
) -> str:
    """
    Return clean Hindi WhatsApp-ready text for a meal plan.

    Args:
        plan: Meal plan dict with days[].meals structure.
        locale: Target locale (hi / hi-IN).

    Returns:
        Formatted Hindi string optimized for WhatsApp.
    """
    english = format_meal_plan_whatsapp_en(plan)
    if not _is_hindi(locale):
        return emoji_safe_format(clean_markdown(english))

    hindi = translate_to_hindi_conversational(english)
    return emoji_safe_format(clean_markdown(hindi))


def translate_to_hindi_conversational(english_text: str) -> str:
    """
    Convert English meal-plan text to conversational Hindi.

    Tries Sarvam Mayura (modern-colloquial) first, then OpenAI.
    """
    if not english_text.strip():
        return ""

    if _sarvam_available():
        try:
            protected, num_map = preserve_numeric_tokens(english_text)
            protected, emoji_map = preserve_emoji_tokens(protected)
            translated = _translate_via_sarvam(protected, mode="modern-colloquial")
            restored = _restore_emoji_tokens(
                restore_numeric_tokens(translated, num_map),
                emoji_map,
            )
            logger.info("Hindi localization via Sarvam (%d chars)", len(restored))
            return restored
        except Exception as exc:
            logger.warning("Sarvam translation failed: %s; falling back to OpenAI", exc)

    # OpenAI prompt instructs preserving numbers/emojis — no placeholder swap needed
    restored = _translate_via_openai(english_text)
    logger.info("Hindi localization via OpenAI (%d chars)", len(restored))
    return restored


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_meal_plan_whatsapp_en(plan: dict[str, Any]) -> str:
    """
    Build structured English WhatsApp text preserving meals, calories, and emojis.

    Args:
        plan: Meal plan with days[].meals.{breakfast,lunch,dinner}.

    Returns:
        Multi-line English string.
    """
    lines: list[str] = ["🍽 *Your Meal Plan*", ""]

    summary = plan.get("summary") or plan.get("content")
    if summary and isinstance(summary, str) and len(summary) < 400:
        lines.append(summary.strip())
        lines.append("")

    score = plan.get("score")
    if score is not None:
        lines.append(f"📊 Plan score: {score}/100")
        lines.append("")

    days = plan.get("days", [])
    if not days:
        fallback = plan.get("content") or plan.get("message") or ""
        return truncate_whatsapp(emoji_safe_format(str(fallback)))

    for day_block in days[:7]:
        day_en = str(day_block.get("day", "")).lower()
        day_label = day_en.capitalize() if day_en else "Today"
        lines.append(f"📅 *{day_label}*")

        meals = day_block.get("meals") or {}
        for meal_type in ("breakfast", "lunch", "dinner", "snack"):
            meal = meals.get(meal_type)
            if not isinstance(meal, dict):
                continue
            lines.extend(_format_meal_line_en(meal_type, meal))

        lines.append("")

    blocked = plan.get("blocked_ingredients") or []
    if blocked:
        lines.append(f"🚫 Avoiding: {', '.join(blocked[:8])}")

    text = "\n".join(lines).strip()
    return truncate_whatsapp(emoji_safe_format(clean_markdown(text)))


def _format_meal_line_en(meal_type: str, meal: dict[str, Any]) -> list[str]:
    """Format a single meal slot for WhatsApp."""
    emoji = MEAL_EMOJI.get(meal_type, "🍽")
    label = meal_type.capitalize()
    name = meal.get("name") or meal.get("title") or "Meal"
    parts = [f"  {emoji} *{label}:* {name}"]

    cal = _extract_calories(meal)
    if cal is not None:
        parts.append(f"     🔥 {cal:.0f} kcal")

    macros = []
    if meal.get("protein") is not None:
        macros.append(f"P {meal['protein']:.0f}g")
    if meal.get("carbs") is not None:
        macros.append(f"C {meal['carbs']:.0f}g")
    if meal.get("fat") is not None:
        macros.append(f"F {meal['fat']:.0f}g")
    if macros:
        parts.append(f"     📊 {' | '.join(macros)}")

    if meal.get("diabetes_score") is not None:
        parts.append(f"     ✅ Score: {meal['diabetes_score']:.2f}")

    short = meal.get("summary", "")
    if short and len(short) < 80:
        parts.append(f"     💬 {short[:80]}")

    return parts


def clean_markdown(text: str) -> str:
    """
    Strip markdown artifacts while keeping line structure and emojis.

    Removes **, __, ##, ```, and HTML tags.
    """
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"```[a-z]*\n?", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\*([^*\n]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def emoji_safe_format(text: str) -> str:
    """
    Normalize whitespace for WhatsApp without breaking emoji clusters.

    - Preserves newlines
    - Collapses horizontal spaces
    - Ensures emoji are followed by space when glued to Latin/Devanagari
    """
    if not text:
        return ""

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = _EMOJI_PATTERN.sub(lambda m: m.group(0) + " ", line)
        line = re.sub(r" +", " ", line).strip()
        lines.append(line)

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def preserve_numeric_tokens(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace numbers and kcal patterns with placeholders for safe translation.

    Returns:
        (protected_text, restore_map)
    """
    restore_map: dict[str, str] = {}
    counter = 0

    def _replacer(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"⟨N{counter}⟩"
        restore_map[token] = match.group(0)
        counter += 1
        return token

    # Most specific patterns first to avoid partial overlaps
    patterns = [
        r"\d+\.?\d*\s*kcal",
        r"\d+\.?\d*\s*(?:g|mg/dL|%)",
        r"Score:\s*\d+\.?\d*",
        r"\d+\.?\d*/100",
        r"\d+\.?\d*",
    ]
    protected = text
    for pattern in patterns:
        protected = re.sub(pattern, _replacer, protected, flags=re.IGNORECASE)

    return protected, restore_map


def restore_numeric_tokens(text: str, restore_map: dict[str, str]) -> str:
    """Restore placeholders to original numeric strings."""
    result = text
    for token, original in restore_map.items():
        result = result.replace(token, original)
    return result


def preserve_emoji_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Replace emojis with placeholders during machine translation."""
    store: dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:
        key = f"@@EMOJI{len(store)}@@"
        store[key] = match.group(0)
        return key

    return _EMOJI_PATTERN.sub(_repl, text), store


def _restore_emoji_tokens(text: str, store: dict[str, str]) -> str:
    """Restore emoji placeholders."""
    result = text
    for key, emoji in store.items():
        result = result.replace(key, emoji)
    return result


def truncate_whatsapp(text: str, max_chars: int = _WHATSAPP_MAX_CHARS) -> str:
    """Truncate to WhatsApp-safe length at line boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 20]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return cut.strip() + "\n\n… (संक्षिप्त)"


# ---------------------------------------------------------------------------
# Sarvam API
# ---------------------------------------------------------------------------


def _sarvam_available() -> bool:
    settings = get_settings()
    key = settings.sarvam_api_key or ""
    return bool(key and "your" not in key.lower() and "changeme" not in key.lower())


def _translate_via_sarvam(text: str, *, mode: str = "modern-colloquial") -> str:
    """
    Translate text to Hindi via Sarvam Mayura API.

    Chunks input to respect API limits (~1000 chars for mayura:v1).
    """
    settings = get_settings()
    chunks = _chunk_text(text, settings.sarvam_max_chunk_size)
    translated_parts: list[str] = []

    with httpx.Client(timeout=60.0) as client:
        for chunk in chunks:
            payload = {
                "input": chunk,
                "source_language_code": "en-IN",
                "target_language_code": "hi-IN",
                "mode": mode,
                "model": settings.sarvam_translate_model,
                "numerals_format": "international",
            }
            headers = {
                "api-subscription-key": settings.sarvam_api_key,
                "Content-Type": "application/json",
            }
            response = client.post(
                settings.sarvam_translate_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            part = (
                data.get("translated_text")
                or data.get("output")
                or data.get("translation")
                or ""
            )
            if not part and isinstance(data.get("data"), dict):
                part = data["data"].get("translated_text", "")
            translated_parts.append(str(part).strip())

    return "\n".join(translated_parts)


# ---------------------------------------------------------------------------
# OpenAI fallback
# ---------------------------------------------------------------------------


def _openai_available() -> bool:
    settings = get_settings()
    key = settings.openai_api_key or ""
    return bool(key and not key.startswith("sk-your") and "your-openai" not in key.lower())


def _translate_via_openai(text: str) -> str:
    """Conversational Hindi via OpenAI."""
    settings = get_settings()
    if not _openai_available():
        return _deterministic_hindi_stub(text)

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": HINDI_LOCALIZATION_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=1800,
    )
    return response.choices[0].message.content or _deterministic_hindi_stub(text)


def _translate_short_text(text: str, *, context: str = "") -> str:
    """Translate a short recipe string."""
    if not text.strip():
        return ""
    if _sarvam_available():
        try:
            return _translate_via_sarvam(text, mode="modern-colloquial")
        except Exception:
            pass
    if _openai_available():
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": HINDI_RECIPE_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content or text
    return text


def _deterministic_hindi_stub(english_text: str) -> str:
    """Minimal Hindi labels when no API is configured."""
    result = english_text
    for en, hi in {**WEEKDAY_EN_TO_HI, **MEAL_TYPE_EN_TO_HI}.items():
        result = re.sub(rf"\b{en}\b", hi, result, flags=re.IGNORECASE)
    replacements = {
        "Your Meal Plan": "आपकी भोजन योजना",
        "Breakfast:": "नाश्ता:",
        "Lunch:": "दोपहर का भोजन:",
        "Dinner:": "रात का खाना:",
        "Avoiding:": "बचें:",
        "Plan score:": "योजना स्कोर:",
        "kcal": "कैलोरी",
    }
    for en, hi in replacements.items():
        result = result.replace(en, hi)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_hindi(locale: str) -> bool:
    return locale.lower().startswith("hi")


def _extract_calories(meal: dict[str, Any]) -> float | None:
    if meal.get("calories") is not None:
        return float(meal["calories"])
    return None


def _chunk_text(text: str, max_length: int) -> list[str]:
    """Split text on word boundaries for API chunk limits."""
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        split_at = remaining.rfind(" ", 0, max_length)
        if split_at <= 0:
            split_at = max_length
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _localize_plan_days_structure(plan: dict[str, Any], locale: str) -> list[dict[str, Any]]:
    """Add Hindi day/meal labels alongside English structure."""
    localized_days = []
    for day_block in plan.get("days", []):
        day_en = str(day_block.get("day", "")).lower()
        entry = {
            "day": day_block.get("day"),
            "day_hi": WEEKDAY_EN_TO_HI.get(day_en, day_block.get("day")),
            "meals": {},
        }
        for meal_type, meal in (day_block.get("meals") or {}).items():
            if isinstance(meal, dict):
                entry["meals"][meal_type] = localize_recipe(meal, locale)
            else:
                entry["meals"][meal_type] = meal
        localized_days.append(entry)
    return localized_days
