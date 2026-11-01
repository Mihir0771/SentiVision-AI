"""
Database seeder script for SentiVision AI.
Populates MongoDB with a demo administrative user and realistic sample analyses.

Usage:
    python scripts/seed_db.py [--clear]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import logging
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymongo import MongoClient
from app.config import config
from app.models.analysis import Analysis
from app.models.user import User

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DATA = [
    {
        "type": "text",
        "input": "This is without a doubt the finest mechanical keyboard I have ever typed on. The switches feel crisp, the RGB illumination is elegant, and the aluminum frame feels virtually indestructible.",
        "sentiment": "positive",
        "confidence": 0.96,
        "models": ["transformer", "vader", "textblob"],
        "keywords": ["mechanical keyboard", "switches", "aluminum frame", "RGB illumination"],
    },
    {
        "type": "text",
        "input": "Appalling service. Flight was delayed by 7 hours with zero communication, customer desk staff was downright rude, and our luggage was lost at the connection hub. Never flying with them again.",
        "sentiment": "negative",
        "confidence": 0.98,
        "models": ["transformer", "vader", "textblob"],
        "keywords": ["flight delayed", "zero communication", "lost luggage", "customer desk"],
    },
    {
        "type": "text",
        "input": "The standard subscription tier includes 50GB cloud storage, weekly automated backups, and access to the web portal. Billing occurs on the 1st of every calendar month.",
        "sentiment": "neutral",
        "confidence": 0.88,
        "models": ["transformer", "vader", "textblob"],
        "keywords": ["subscription tier", "cloud storage", "automated backups", "calendar month"],
    },
    {
        "type": "facial",
        "input": "Live camera capture — User reaction during product onboarding walkthrough",
        "sentiment": "happy",
        "confidence": 0.92,
        "models": ["fer_facial_model"],
        "keywords": ["webcam stream", "happy expression"],
    },
    {
        "type": "facial",
        "input": "Live camera snapshot — Student focus and engagement check during lecture",
        "sentiment": "surprise",
        "confidence": 0.85,
        "models": ["fer_facial_model"],
        "keywords": ["facial tracking", "surprise expression"],
    },
    {
        "type": "batch",
        "input": "Batch File: amazon_audio_reviews_q3.csv (450 rows processed)",
        "sentiment": "positive",
        "confidence": 0.87,
        "models": ["ensemble"],
        "keywords": ["amazon reviews", "bulk processing", "audio accessories"],
    },
    {
        "type": "url",
        "input": "https://reddit.com/r/technology/comments/new_release_discussion",
        "sentiment": "positive",
        "confidence": 0.79,
        "models": ["ensemble", "scraper"],
        "keywords": ["reddit discussion", "scraped comments", "technology thread"],
    },
]


def seed_database(clear_first: bool = False) -> None:
    """Connect to MongoDB and seed user and analyses collections."""
    cfg = config["development"]
    mongo_uri = os.environ.get("MONGO_URI", cfg.MONGO_URI)

    logger.info("Connecting to MongoDB at: %s", mongo_uri)
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        db_name = mongo_uri.rsplit("/", 1)[-1].split("?")[0] or "sentiment_analyzer"
        db = client[db_name]
        logger.info("Successfully connected to database '%s'", db_name)
    except Exception as exc:
        logger.error("Could not establish connection to MongoDB: %s", exc)
        logger.info("Ensure MongoDB service is running (e.g. `docker-compose up mongo` or local mongod).")
        return

    if clear_first:
        logger.info("Clearing existing users and analyses collections…")
        db["users"].delete_many({})
        db["analyses"].delete_many({})

    # 1. Create or update Default Admin User
    admin_user = User.find_by_username(db, "admin")
    if not admin_user:
        admin_user = User(username="admin", email="admin@sentivision.ai")
        admin_user.set_password("admin123")
        admin_id = admin_user.save(db)
        logger.info("Created administrator user: username='admin', password='admin123' (id=%s)", admin_id)
    else:
        admin_id = admin_user.id
        logger.info("Administrator user already present (id=%s)", admin_id)

    # 2. Insert sample analysis records with staggered timestamps
    now = datetime.now(timezone.utc)
    inserted = 0

    for i, item in enumerate(SAMPLE_DATA * 3):  # 21 items total
        ts = now - timedelta(days=i // 3, hours=(i % 3) * 6)
        analysis = Analysis(
            user_id=admin_id,
            analysis_type=item["type"],
            input_data=item["input"],
            results={
                "overall_sentiment": item["sentiment"],
                "overall_confidence": item["confidence"],
                "ensemble_scores": {
                    "positive": item["confidence"] if item["sentiment"] == "positive" else 0.1,
                    "negative": item["confidence"] if item["sentiment"] == "negative" else 0.1,
                    "neutral": item["confidence"] if item["sentiment"] == "neutral" else 0.1,
                },
                "models": {m: {"label": item["sentiment"]} for m in item["models"]},
            },
            overall_sentiment=item["sentiment"],
            confidence=item["confidence"],
            models_used=item["models"],
            keywords=item["keywords"],
            created_at=ts,
        )
        analysis.save(db)
        inserted += 1

    logger.info("Seeded %d sample analyses into MongoDB successfully!", inserted)
    logger.info("Ready! Start server with: python run.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed SentiVision AI database.")
    parser.add_argument("--clear", action="store_true", help="Clear collections before seeding.")
    args = parser.parse_args()

    seed_database(clear_first=args.clear)
