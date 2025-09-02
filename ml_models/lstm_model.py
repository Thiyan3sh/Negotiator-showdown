import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, Tuple
import logging

class LSTMSequencePredictor:
    """LSTM model for predicting next offers in negotiation sequences"""

    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """Initialize LSTM predictor"""
        self.config = config or {
            "units": 64, "dropout": 0.2, "epochs": 50, "batch_size": 32,
            "sequence_length": 5
        }
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.logger = logger or logging.getLogger("LSTMSequencePredictor")

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training"""
        sequences, targets = [], []
        negotiation_groups = df.groupby(["product", "buyer_personality", "seller_personality"])
        for _, group in negotiation_groups:
            if len(group) < self.config["sequence_length"] + 1:
                continue
            group = group.sort_values("round")
            offers = group["current_offer"].values
            offers_scaled = self.scaler.fit_transform(offers.reshape(-1, 1)).flatten()
            for i in range(len(offers_scaled) - self.config["sequence_length"]):
                sequences.append(offers_scaled[i:i + self.config["sequence_length"]])
                targets.append(offers_scaled[i + self.config["sequence_length"]])
        X = np.array(sequences)
        y = np.array(targets)
        return X.reshape((X.shape[0], X.shape[1], 1)), y

    def train(self, df: pd.DataFrame) -> Dict:
        """Train LSTM model"""
        self.logger.info("Training LSTM sequence prediction model")
        X, y = self.prepare_sequences(df)
        if len(X) == 0:
            self.logger.error("No sequence data available for training")
            return {"success": False, "error": "No sequence data"}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = Sequential([
            LSTM(self.config["units"], return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(self.config["dropout"]),
            LSTM(self.config["units"] // 2, return_sequences=False),
            Dropout(self.config["dropout"]),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        history = self.model.fit(X_train, y_train, epochs=self.config["epochs"],
                                 batch_size=self.config["batch_size"],
                                 validation_data=(X_test, y_test), verbose=0)
        self.is_trained = True
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)[0]
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)[0]
        self.logger.info(f"LSTM training complete. Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
        return {"success": True, "train_loss": train_loss, "test_loss": test_loss, "history": history.history}

    def predict(self, sequence: np.ndarray) -> float:
        """Predict next offer in sequence"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1, 1)
        elif len(sequence.shape) == 2:
            sequence = sequence.reshape(sequence.shape[0], sequence.shape[1], 1)
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, 1)).reshape(sequence.shape)
        prediction_scaled = self.model.predict(sequence_scaled, verbose=0)
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        return prediction[0][0]

    def save_model(self, filepath: str):
        """Save trained model"""
        if self.is_trained:
            self.model.save(f"{filepath}_lstm.h5")
            joblib.dump({"scaler": self.scaler, "config": self.config}, f"{filepath}_lstm_utils.pkl")
            self.logger.info(f"LSTM model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            self.model = load_model(f"{filepath}_lstm.h5")
            utils = joblib.load(f"{filepath}_lstm_utils.pkl")
            self.scaler = utils["scaler"]
            self.config = utils["config"]
            self.is_trained = True
            self.logger.info(f"LSTM model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
