"""Utilities for ML Model Management"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from ml_models.xgboost_model import XGBoostPredictor
from ml_models.lstm_model import LSTMSequencePredictor
from ml_models.sarima_model import SARIMAForecaster
from ml_models.dataset_generator import DatasetGenerator

class ModelManager:
    """Manages training, loading, and coordination of all ML models"""
    
    def __init__(self, model_dir: str, data_dir: str):
        """Initialize model manager"""
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.models = {
            "xgboost": XGBoostPredictor(),
            "lstm": LSTMSequencePredictor(),
            "sarima": SARIMAForecaster()
        }
        self.logger = logging.getLogger(__name__)
        
    def generate_training_data(self, num_negotiations: int = 5000) -> Dict[str, pd.DataFrame]:
        """Generate synthetic training datasets"""
        self.logger.info(f"Generating {num_negotiations} synthetic negotiations")
        
        generator = DatasetGenerator(num_negotiations)
        negotiation_df = generator.generate_negotiation_dataset()
        timeseries_df = generator.generate_time_series_data()
        
        # Save datasets
        generator.save_datasets(negotiation_df, timeseries_df, self.data_dir)
        
        return {
            "negotiation": negotiation_df,
            "timeseries": timeseries_df
        }
    
    def train_all_models(self, datasets: Optional[Dict] = None) -> Dict:
        """Train all ML models"""
        if datasets is None:
            datasets = self.generate_training_data()
        
        training_results = {}
        
        # Train XGBoost
        try:
            xgb_result = self.models["xgboost"].train(datasets["negotiation"])
            training_results["xgboost"] = xgb_result
            if xgb_result["success"]:
                self.models["xgboost"].save_model(f"{self.model_dir}/xgboost_model")
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            training_results["xgboost"] = {"success": False, "error": str(e)}
        
        # Train LSTM
        try:
            lstm_result = self.models["lstm"].train(datasets["negotiation"])
            training_results["lstm"] = lstm_result
            if lstm_result["success"]:
                self.models["lstm"].save_model(f"{self.model_dir}/lstm_model")
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            training_results["lstm"] = {"success": False, "error": str(e)}
        
        # Train SARIMA
        try:
            sarima_result = self.models["sarima"].train(datasets["timeseries"])
            training_results["sarima"] = sarima_result
            if sarima_result["success"]:
                self.models["sarima"].save_model(f"{self.model_dir}/sarima_model")
        except Exception as e:
            self.logger.error(f"SARIMA training failed: {e}")
            training_results["sarima"] = {"success": False, "error": str(e)}
        
        return training_results
    
    def load_models(self) -> Dict:
        """Load all trained models"""
        loading_results = {}
        
        for model_name, model in self.models.items():
            try:
                model.load_model(f"{self.model_dir}/{model_name}_model")
                loading_results[model_name] = {"success": True}
            except Exception as e:
                self.logger.warning(f"Could not load {model_name} model: {e}")
                loading_results[model_name] = {"success": False, "error": str(e)}
        
        return loading_results
    
    def get_models_for_agent(self) -> Dict:
        """Get trained models for agent use"""
        return {
            name: model if model.is_trained else None 
            for name, model in self.models.items()
        }
