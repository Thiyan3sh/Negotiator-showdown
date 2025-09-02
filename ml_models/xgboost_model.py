# ml_models/xgboost_model.py
"""XGBoost Model for Price Prediction"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
from typing import Dict, Tuple, Optional
import logging

class XGBoostPredictor:
    """XGBoost model for predicting optimal negotiation prices"""
    
    def __init__(self, config: Dict = None, logger: Optional[logging.Logger] = None):
        """Initialize XGBoost predictor"""
        self.config = config or {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.logger = logger or logging.getLogger("XGBoostPredictor")
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        for col in ["product", "buyer_personality", "seller_personality"]:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col])
        
        df["budget_to_min_ratio"] = df["buyer_budget"] / df["seller_min_price"]
        df["price_position"] = (df["current_offer"] - df["seller_min_price"]) / (df["buyer_budget"] - df["seller_min_price"])
        df["round_progress"] = df["round"] / 10
        df["utility_gap"] = df["buyer_utility"] - df["seller_utility"]
        
        feature_cols = [
            "product_encoded", "round", "buyer_personality_encoded", "seller_personality_encoded",
            "buyer_budget", "seller_min_price", "price_range", "current_offer",
            "buyer_utility", "seller_utility", "time_pressure",
            "budget_to_min_ratio", "price_position", "round_progress", "utility_gap"
        ]
        
        self.feature_columns = feature_cols
        X = df[feature_cols].fillna(0)
        successful_negotiations = df[df["success"] == True]
        y = successful_negotiations["final_price"].fillna(0)
        X_filtered = X.loc[successful_negotiations.index]
        
        return X_filtered.values, y.values
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train XGBoost model"""
        self.logger.info("Training XGBoost price prediction model")
        X, y = self.prepare_features(df)
        if len(X) == 0:
            self.logger.error("No training data available")
            return {"success": False, "error": "No training data"}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = xgb.XGBRegressor(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            random_state=42
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.logger.info(f"XGBoost training complete. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        return {
            "success": True,
            "train_score": train_score,
            "test_score": test_score,
            "feature_importance": dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(features)
    
    def save_model(self, filepath: str):
        if self.is_trained:
            model_data = {
                "model": self.model,
                "label_encoders": self.label_encoders,
                "feature_columns": self.feature_columns,
                "config": self.config
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data["model"]
            self.label_encoders = model_data["label_encoders"]
            self.feature_columns = model_data["feature_columns"]
            self.config = model_data["config"]
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
