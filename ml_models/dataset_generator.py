"""Synthetic Dataset Generator for Training ML Models"""
import sys
import os
import csv
import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PRODUCTS, PERSONALITY_PARAMS

class DatasetGenerator:
    """Generate synthetic negotiation datasets for ML training"""

    def __init__(self, num_negotiations: int = 10000):
        self.num_negotiations = num_negotiations
        self.products = PRODUCTS
        self.personalities = list(PERSONALITY_PARAMS.keys())
        os.makedirs("data", exist_ok=True)

    def generate_negotiation_dataset(self) -> pd.DataFrame:
        """Generate comprehensive negotiation dataset"""
        data = []

        for _ in range(self.num_negotiations):
            product_name = random.choice(list(self.products.keys()))
            product_info = self.products[product_name]

            buyer_personality = random.choice(self.personalities)
            seller_personality = random.choice(self.personalities)

            base_price = random.uniform(product_info["min_price"], product_info["max_price"])
            buyer_budget = base_price * random.uniform(0.8, 2.0)
            seller_min_price = base_price * random.uniform(0.3, 1.5)

            negotiation = self._simulate_negotiation(
                product_name, buyer_personality, seller_personality,
                buyer_budget, seller_min_price, product_info
            )
            data.extend(negotiation)

        df = pd.DataFrame(data)
        return df

    def _simulate_negotiation(self, product: str, buyer_personality: str, 
                              seller_personality: str, buyer_budget: float, 
                              seller_min_price: float, product_info: Dict) -> List[Dict]:
        """Simulate a complete negotiation"""
        rounds_data = []
        current_price = None
        success = False

        buyer_traits = PERSONALITY_PARAMS[buyer_personality]
        seller_traits = PERSONALITY_PARAMS[seller_personality]

        max_rounds = random.randint(1, 10)

        for round_num in range(1, max_rounds + 1):
            if round_num == 1:
                # Seller initial offer
                if seller_personality == "aggressive":
                    current_price = product_info["max_price"] * random.uniform(1.2, 2.0)
                elif seller_personality == "diplomatic":
                    current_price = (product_info["min_price"] + product_info["max_price"]) * 0.6
                elif seller_personality == "analytical":
                    current_price = product_info["max_price"] * random.uniform(0.8, 1.2)
                else:  # wildcard
                    current_price = product_info["max_price"] * random.uniform(0.5, 1.8)
            else:
                if round_num % 2 == 0:  # Buyer's turn
                    concession = buyer_traits["concession_rate"] * random.uniform(0.5, 1.5)
                    current_price *= (1 - concession if buyer_personality != "aggressive" else 1 - concession * 0.5)
                    current_price = max(current_price, seller_min_price * 0.9)
                else:  # Seller's turn
                    concession = seller_traits["concession_rate"] * random.uniform(0.5, 1.5)
                    current_price *= (1 - concession)
                    current_price = max(current_price, seller_min_price)

            if (current_price <= buyer_budget and current_price >= seller_min_price and
                random.random() < self._calculate_success_probability(round_num, buyer_personality, seller_personality)):
                success = True
                break

            round_data = {
                "product": product,
                "round": round_num,
                "current_offer": current_price,
                "buyer_personality": buyer_personality,
                "seller_personality": seller_personality,
                "buyer_budget": buyer_budget,
                "seller_min_price": seller_min_price,
                "price_range": product_info["max_price"] - product_info["min_price"],
                "buyer_utility": self._calculate_utility(current_price, buyer_budget, "buyer"),
                "seller_utility": self._calculate_utility(current_price, seller_min_price, "seller", product_info["max_price"]),
                "time_pressure": round_num / 10,
                "success": success,
                "final_price": current_price if success else None
            }
            rounds_data.append(round_data)

        return rounds_data

    def _calculate_success_probability(self, round_num: int, buyer_personality: str, seller_personality: str) -> float:
        base_prob = 0.1 + (round_num * 0.08)
        compatibility_bonus = 0
        if buyer_personality == "diplomatic" or seller_personality == "diplomatic":
            compatibility_bonus += 0.2
        if buyer_personality == "aggressive" and seller_personality == "aggressive":
            compatibility_bonus -= 0.1
        return min(0.9, base_prob + compatibility_bonus)

    def _calculate_utility(self, price: float, constraint: float, role: str, max_price: float = None) -> float:
        if role == "buyer":
            return -1 if price > constraint else (constraint - price) / constraint
        else:
            return -1 if price < constraint else (price - constraint) / (max_price - constraint) if max_price else 0.5

    def generate_time_series_data(self, sequence_length: int = 1000) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=sequence_length, freq="D")
        time_series_data = []

        for product in self.products.keys():
            product_info = self.products[product]
            base_price = (product_info["min_price"] + product_info["max_price"]) / 2

            trend = np.cumsum(np.random.normal(0, 0.02, sequence_length))
            seasonal = 10 * np.sin(2 * np.pi * np.arange(sequence_length) / 365.25)
            noise = np.random.normal(0, 1, sequence_length)

            prices = base_price + trend + seasonal + noise
            prices = np.clip(prices, product_info["min_price"], product_info["max_price"])

            for i, (date, price) in enumerate(zip(dates, prices)):
                time_series_data.append({
                    "date": date,
                    "product": product,
                    "price": price,
                    "day_of_year": date.dayofyear,
                    "month": date.month,
                    "quarter": (date.month - 1) // 3 + 1,
                    "trend": trend[i],
                    "seasonal": seasonal[i]
                })

        return pd.DataFrame(time_series_data)

    def save_datasets(self, negotiation_df: pd.DataFrame, timeseries_df: pd.DataFrame, data_dir: str):
        negotiation_df.to_csv(f"{data_dir}/negotiation_data.csv", index=False)
        timeseries_df.to_csv(f"{data_dir}/price_timeseries.csv", index=False)
        metadata = {
            "negotiation_records": len(negotiation_df),
            "timeseries_records": len(timeseries_df),
            "products": list(self.products.keys()),
            "personalities": self.personalities,
            "generation_date": pd.Timestamp.now().isoformat()
        }
        with open(f"{data_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
