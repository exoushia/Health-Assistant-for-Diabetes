"""
MongoDB upload helpers — future migration path, not used in the live copilot yet.

Functions:
    get_mongo_client              — lazy pymongo client from MONGODB_URI
    upload_recipes_to_mongodb     — bulk insert enriched recipes
    upload_users_to_mongodb       — bulk insert user profiles
    upload_enriched_file_to_mongodb — load JSON file then upload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_mongo_client():
    """
    Return a pymongo MongoClient from configured URI.

    Raises:
        ImportError: If pymongo is not installed.
        ValueError: If MONGODB_URI is empty.
    """
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise ImportError("pymongo is required for MongoDB uploads") from exc

    settings = get_settings()
    if not settings.mongodb_uri:
        raise ValueError("MONGODB_URI is not configured")
    return MongoClient(settings.mongodb_uri)


def upload_recipes_to_mongodb(
    recipes: list[dict[str, Any]],
    *,
    collection_name: str = "recipes",
    upsert: bool = True,
) -> dict[str, Any]:
    """
    Upload enriched recipes to MongoDB.

    Uses recipe_id as the unique key when upsert=True.

    Args:
        recipes: List of recipe dicts (RecipeSchema-compatible).
        collection_name: Target collection name.
        upsert: Replace existing documents with matching recipe_id.

    Returns:
        Summary dict with inserted, modified, and matched counts.
    """
    settings = get_settings()
    client = get_mongo_client()
    collection = client[settings.mongodb_db_name][collection_name]

    inserted = 0
    modified = 0
    matched = 0

    for recipe in recipes:
        recipe_id = recipe.get("recipe_id")
        if not recipe_id:
            logger.warning("Skipping recipe without recipe_id")
            continue
        if upsert:
            result = collection.replace_one({"recipe_id": recipe_id}, recipe, upsert=True)
            if result.matched_count:
                matched += 1
            if result.modified_count:
                modified += 1
            if result.upserted_id:
                inserted += 1
        else:
            collection.insert_one(recipe)
            inserted += 1

    client.close()
    summary = {
        "collection": collection_name,
        "database": settings.mongodb_db_name,
        "total": len(recipes),
        "inserted": inserted,
        "modified": modified,
        "matched": matched,
    }
    logger.info("MongoDB recipe upload: %s", summary)
    return summary


def upload_users_to_mongodb(
    users_path: str | Path | None = None,
    *,
    collection_name: str = "users",
    upsert: bool = True,
) -> dict[str, Any]:
    """
    Upload users from data/users.json to MongoDB.

    Args:
        users_path: Path to users JSON file (default from settings).
        collection_name: Target collection name.
        upsert: Replace existing documents with matching id.

    Returns:
        Summary dict with upload counts.
    """
    settings = get_settings()
    path = Path(users_path or settings.users_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Users file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    users = payload.get("users", payload if isinstance(payload, list) else [])

    client = get_mongo_client()
    collection = client[settings.mongodb_db_name][collection_name]

    inserted = 0
    modified = 0

    for user in users:
        user_id = user.get("id") or user.get("user_id")
        if not user_id:
            logger.warning("Skipping user without id")
            continue
        if upsert:
            result = collection.replace_one({"id": user_id}, user, upsert=True)
            if result.modified_count:
                modified += 1
            if result.upserted_id:
                inserted += 1
        else:
            collection.insert_one(user)
            inserted += 1

    client.close()
    summary = {
        "collection": collection_name,
        "database": settings.mongodb_db_name,
        "total": len(users),
        "inserted": inserted,
        "modified": modified,
    }
    logger.info("MongoDB user upload: %s", summary)
    return summary


def upload_enriched_file_to_mongodb(
    enriched_path: str | Path,
    *,
    collection_name: str = "recipes",
) -> dict[str, Any]:
    """
    Load enriched_recipes.json and upload all recipes to MongoDB.

    Convenience wrapper for one-shot migration.
    """
    path = Path(enriched_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    recipes = data.get("recipes", data if isinstance(data, list) else [])
    return upload_recipes_to_mongodb(recipes, collection_name=collection_name)
