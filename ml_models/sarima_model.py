"""SARIMA Model for Price Trend Forecasting"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import joblib
from typing import Dict, Tuple, Optional, List
import logging
import warnings
warnings.filterwarnings('ignore')

class SARIMAForecaster:
    """SARIMA model for price trend forecasting"""
    
    def __init__(self, config: Dict = None, logger: Optional[logging.Logger] = None):
        """Initialize SARIMA forecaster"""
        self.config = config or {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)}
        self.models = {}  # One model per product
        self.scalers = {}
        self.is_trained = False
        self.logger = logger or logging.getLogger("SARIMAForecaster")

    def prepare_timeseries(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Prepare time series data for each product"""
        timeseries_data = {}
        
        for product in df["product"].unique():
            product_data = df[df["product"] == product].copy()
            product_data = product_data.sort_values("date")
            product_data.set_index("date", inplace=True)
            
            # Ensure we have enough data points
            if len(product_data) < 50:
                self.logger.warning(f"Insufficient data for {product}: {len(product_data)} points")
                continue
            
            timeseries_data[product] = product_data["price"]
        
        return timeseries_data
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train SARIMA models for each product"""
        self.logger.info("Training SARIMA forecasting models")
        
        timeseries_data = self.prepare_timeseries(df)
        training_results = {}
        
        for product, series in timeseries_data.items():
            try:
                # Check stationarity
                adf_test = adfuller(series)
                is_stationary = adf_test[1] < 0.05
                
                if not is_stationary:
                    # Difference the series
                    series_diff = series.diff().dropna()
                    if len(series_diff) < 30:
                        continue
                    training_series = series_diff
                else:
                    training_series = series
                
                # Fit SARIMA model
                model = ARIMA(
                    training_series,
                    order=self.config["order"],
                    seasonal_order=self.config["seasonal_order"]
                )
                
                fitted_model = model.fit()
                self.models[product] = {
                    "model": fitted_model,
                    "original_series": series,
                    "differenced": not is_stationary,
                    "last_values": series.tail(self.config["order"][1] + 1).values if not is_stationary else None
                }
                
                # Calculate model performance
                aic = fitted_model.aic
                bic = fitted_model.bic
                
                training_results[product] = {
                    "success": True,
                    "aic": aic,
                    "bic": bic,
                    "stationary": is_stationary
                }
                
                self.logger.info(f"SARIMA model trained for {product}. AIC: {aic:.2f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train SARIMA for {product}: {e}")
                training_results[product] = {"success": False, "error": str(e)}
        
        self.is_trained = len(self.models) > 0
        
        return {
            "success": self.is_trained,
            "models_trained": len(self.models),
            "results": training_results
        }
    
    def forecast(self, product: str, steps: int = 1) -> np.ndarray:
        """Forecast future prices for a product"""
        if not self.is_trained or product not in self.models:
            raise ValueError(f"Model for {product} not trained")
        
        model_info = self.models[product]
        fitted_model = model_info["model"]
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=steps)
        
        # If series was differenced, integrate back
        if model_info["differenced"]:

            last_value = model_info["original_series"].iloc[-1]
            integrated_forecast = []
            current_value = last_value
            
            for diff_value in forecast:
                current_value += diff_value
                integrated_forecast.append(current_value)
            
            return np.array(integrated_forecast)
        
        return forecast
    
    def predict_price_direction(self, product: str, current_price: float) -> str:
        """Predict if price will go up, down, or stay stable"""
        try:
            forecast = self.forecast(product, steps=1)[0]
            
            if forecast > current_price * 1.02:
                return "up"
            elif forecast < current_price * 0.98:
                return "down"
            else:
                return "stable"
        except:
            return "stable"
    
    def save_model(self, filepath: str):
        """Save trained models"""
        if self.is_trained:
            model_data = {
                "models": self.models,
                "config": self.config
            }
            joblib.dump(model_data, f"{filepath}_sarima.pkl")
            self.logger.info(f"SARIMA models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models"""
        try:
            model_data = joblib.load(f"{filepath}_sarima.pkl")
            self.models = model_data["models"]
            self.config = model_data["config"]
            self.is_trained = True
            self.logger.info(f"SARIMA models loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load SARIMA models: {e}")

