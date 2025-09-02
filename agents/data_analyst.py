"""Data-Driven Analyst Agent Implementation"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from agents.base_agent import BaseAgent

class DataAnalyst(BaseAgent):
    """Data-driven negotiation using ML predictions and statistical analysis"""
    
    def __init__(self, name, role):
        super().__init__(name, role, "analytical")
        self.confidence_threshold = 0.7
        self.prediction_weight = 0.8
        
    def propose_offer(self, round_num: int, previous_offers: List[float]) -> float:
        """Propose data-driven offer"""
        
        # Collect market data features
        market_features = self._analyze_market_conditions()
        
        if round_num == 1:
            # Use ML model for initial offer if available
            if "xgboost" in self.ml_models and self.ml_models["xgboost"]:
                features = self._prepare_comprehensive_features(round_num, [], market_features)
                predicted_price = self.ml_models["xgboost"].predict(features)[0]
                
                # Adjust based on role and constraint
                if self.role == "buyer":
                    offer = min(predicted_price * 0.9, self.constraint)
                else:
                    offer = max(predicted_price * 1.1, self.constraint)
            else:
                # Fallback to statistical approach
                traits = self.get_personality_traits()
                price_range = self.product_info["max_price"] - self.product_info["min_price"]
                if self.role == "buyer":
                    offer = self.product_info["min_price"] + price_range * traits["initial_offer_factor"]
                else:
                    offer = self.product_info["max_price"] - price_range * (1 - traits["initial_offer_factor"])
        else:
            # Use time series prediction
            if "sarima" in self.ml_models and self.ml_models["sarima"] and len(previous_offers) > 2:
                # Predict next acceptable price
                forecast = self.ml_models["sarima"].forecast(steps=1)
                predicted_price = forecast[0]
                
                # Weight prediction with constraint
                if self.role == "buyer":
                    offer = min(predicted_price * self.prediction_weight + 
                              self.constraint * (1 - self.prediction_weight), 
                              self.constraint)
                else:
                    offer = max(predicted_price * self.prediction_weight + 
                              self.constraint * (1 - self.prediction_weight), 
                              self.constraint)
            else:
                # Statistical analysis of offer patterns
                offer = self._statistical_offer(round_num, previous_offers)
                
        return round(offer, 2)
    
    def respond_offer(self, offer: float, round_num: int) -> Tuple[str, Optional[float]]:
        """Respond based on data analysis"""
        utility = self.calculate_utility(offer)
        
        # Calculate acceptance probability
        accept_prob = self._calculate_acceptance_probability(offer, round_num, utility)
        
        if accept_prob > self.confidence_threshold:
            return ("accept", None)
        
        if not self.is_acceptable(offer):
            # Use ML to find optimal counter
            if "lstm" in self.ml_models and self.ml_models["lstm"]:
                sequence = np.array([offer]).reshape(1, 1, 1)
                predicted_next = self.ml_models["lstm"].predict(sequence)[0][0]
                
                if self.role == "buyer":
                    counter = min(predicted_next, self.constraint)
                else:
			
                    counter = max(predicted_next, self.constraint)
            else:
                counter = self.constraint
            return ("counter", round(counter, 2))
        
        # Analytical counter-offer
        counter = self._analytical_counter(offer, round_num, utility)
        return ("counter", round(counter, 2))
    
    def _analyze_market_conditions(self) -> Dict:
        """Analyze market conditions from historical data"""
        # Simulate market analysis
        return {
            "volatility": np.random.uniform(0.1, 0.3),
            "trend": np.random.choice([-1, 0, 1]),
            "demand": np.random.uniform(0.3, 0.9),
            "supply": np.random.uniform(0.3, 0.9)
        }
    
    def _prepare_comprehensive_features(self, round_num: int, offers: List[float], 
                                       market: Dict) -> np.ndarray:
        """Prepare comprehensive feature set for ML models"""
        features = [
            round_num,
            len(offers),
            np.mean(offers) if offers else 0,
            np.std(offers) if len(offers) > 1 else 0,
            self.constraint,
            1 if self.role == "buyer" else 0,
            market["volatility"],
            market["trend"],
            market["demand"],
            market["supply"]
        ]
        return np.array(features).reshape(1, -1)
    
    def _statistical_offer(self, round_num: int, previous_offers: List[float]) -> float:
        """Generate offer using statistical analysis"""
        if not previous_offers:
            return self.constraint
        
        # Analyze trend
        if len(previous_offers) > 1:
            trend = np.polyfit(range(len(previous_offers)), previous_offers, 1)[0]
            
            # Extrapolate with decay
            decay_factor = 0.9 ** round_num
            next_value = previous_offers[-1] + trend * decay_factor
            
            if self.role == "buyer":
                return min(next_value, self.constraint)
            else:
                return max(next_value, self.constraint)
        else:
            # Simple adjustment
            if self.role == "buyer":
                return min(previous_offers[0] * 1.1, self.constraint)
            else:
                return max(previous_offers[0] * 0.9, self.constraint)
    
    def _calculate_acceptance_probability(self, offer: float, round_num: int, 
                                         utility: float) -> float:
        """Calculate probability of accepting an offer"""
        # Time pressure factor
        time_pressure = round_num / 10  # Max 10 rounds
        
        # Utility factor
        utility_factor = max(0, utility)
        
        # Combined probability
        base_prob = utility_factor * 0.7 + time_pressure * 0.3
        
        # Adjust for market conditions
        if hasattr(self, "_last_market_analysis"):
            if self._last_market_analysis["trend"] > 0 and self.role == "seller":
                base_prob *= 0.9  # Wait for better offers
            elif self._last_market_analysis["trend"] < 0 and self.role == "buyer":
                base_prob *= 1.1  # Accept quickly
                
        return min(1.0, base_prob)
    
    def _analytical_counter(self, offer: float, round_num: int, utility: float) -> float:
        """Generate analytical counter-offer"""
        # Calculate optimal counter based on utility theory
        target_utility = 0.6 - (round_num * 0.05)  # Decrease expectations over time
        
        if self.role == "buyer":
            # Calculate price for target utility
            target_price = self.constraint - (target_utility * self.constraint)
            # Move towards target from current offer
            counter = offer * 0.3 + target_price * 0.7
            return min(counter, self.constraint)
        else:
            # Calculate price for target utility
            max_price = self.product_info["max_price"]
            target_price = self.constraint + (target_utility * (max_price - self.constraint))
            # Move towards target from current offer
            counter = offer * 0.3 + target_price * 0.7
            return max(counter, self.constraint)

