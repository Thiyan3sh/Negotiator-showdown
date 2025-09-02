"""Global configuration for AI Negotiator Showdown"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml_models" / "saved_models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Game Configuration
NEGOTIATION_TIME_LIMIT = 180  # 3 minutes in seconds
MAX_ROUNDS_PER_NEGOTIATION = 10
MIN_OFFER_INCREMENT = 0.01

# Products with min and max price ranges
PRODUCTS = {
    "Laptop": {"min_price": 900, "max_price": 1500, "category": "Electronics"},
    "Phone": {"min_price": 500, "max_price": 800, "category": "Electronics"},
    "Book": {"min_price": 20, "max_price": 50, "category": "Education"},
    "Tablet": {"min_price": 350, "max_price": 600, "category": "Electronics"},
    "Headphones": {"min_price": 100, "max_price": 200, "category": "Accessories"},
}

# Scoring weights
SCORING_WEIGHTS = {
    "profit_savings": 0.4,
    "character_consistency": 0.4,
    "speed_bonus": 0.2
}

# Agent personality parameters (lowercase keys)
PERSONALITY_PARAMS = {
    "aggressivetrader": {
        "concession_rate": 0.1,
        "patience": 0.3,
        "initial_offer_factor": 0.9
    },
    "smoothdiplomat": {
        "concession_rate": 0.05,
        "patience": 0.8,
        "initial_offer_factor": 0.8
    },
    "dataanalyst": {
        "concession_rate": 0.07,
        "patience": 0.5,
        "initial_offer_factor": 0.85
    },
    "wildcard": {
        "concession_rate": 0.15,
        "patience": 0.4,
        "initial_offer_factor": 0.7
    },
    "diplomatic": {  # fallback personality
        "concession_rate": 0.05,
        "patience": 0.8,
        "initial_offer_factor": 0.8
    }
}

# ML Model parameters
ML_CONFIG = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    "lstm": {
        "units": 64,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32
    },
    "sarima": {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 12)
    }
}

# Tournament settings
TOURNAMENT_CONFIG = {
    "round_robin": {
        "rounds_per_match": 2
    },
    "elimination": {
        "best_of": 3
    },
    "grand_finals": {
        "best_of": 5
    }
}
