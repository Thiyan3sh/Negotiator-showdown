"""Aggressive Trader Agent Implementation"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from agents.base_agent import BaseAgent
from .base_agent import BaseAgent

class AggressiveTrader(BaseAgent):
    """Aggressive negotiation style - starts with extreme offers, minimal concessions"""
    
    def __init__(self, name, role):
        super().__init__(name, role, "aggressive")
        self.stubbornness_factor = 0.8
        self.bluff_probability = 0.3
        
    def propose_offer(self, round_num: int, previous_offers: List[float]) -> float:
        """Propose aggressive offer"""
        traits = self.get_personality_traits()
        
        if round_num == 1:
            # Start with extreme offer
            if self.role == "buyer":
                base = self.product_info["min_price"]
                offer = base + (self.constraint - base) * traits["initial_offer_factor"]
            else:
                base = self.product_info["max_price"]
                offer = base - (base - self.constraint) * (1 - traits["initial_offer_factor"])
        else:
            # Make minimal concessions
            last_offer = previous_offers[-1] if previous_offers else self.constraint
            
            # Use ML model if available
            if "xgboost" in self.ml_models and self.ml_models["xgboost"]:
                features = self._prepare_features(round_num, previous_offers)
                ml_suggestion = self.ml_models["xgboost"].predict(features)
                concession = min(traits["concession_rate"], abs(ml_suggestion - last_offer) * 0.5)
            else:
                concession = traits["concession_rate"] * (1 - self.stubbornness_factor)
            
            if self.role == "buyer":
                offer = min(last_offer * (1 + concession), self.constraint)
            else:
                offer = max(last_offer * (1 - concession), self.constraint)
                
        # Occasionally bluff with worse offer
        if np.random.random() < self.bluff_probability and round_num > 2:
            if self.role == "buyer":
                offer *= 0.95
            else:
                offer *= 1.05
                
        return round(offer, 2)
    
    def respond_offer(self, offer: float, round_num: int) -> Tuple[str, Optional[float]]:
        """Respond aggressively to offers"""
        utility = self.calculate_utility(offer)
        
        # Accept only very good offers
        if utility > 0.7:
            return ("accept", None)
        
        # Reject bad offers outright
        if utility < 0.2 or not self.is_acceptable(offer):
            if round_num < 8:  # Still time to negotiate
                counter = self.propose_offer(round_num, [offer])
                return ("counter", counter)
            else:
                # Near deadline, might accept marginal offer
                if utility > 0:
                    return ("accept", None)
                return ("reject", None)
        
        # Counter with aggressive offer
        counter = self.propose_offer(round_num, [offer])
        return ("counter", counter)
    
    def _prepare_features(self, round_num: int, previous_offers: List[float]) -> np.ndarray:
        """Prepare features for ML model"""
        features = [
            round_num,
            len(previous_offers),
            np.mean(previous_offers) if previous_offers else 0,
            np.std(previous_offers) if len(previous_offers) > 1 else 0,
            self.constraint,
            1 if self.role == "buyer" else 0
        ]
        return np.array(features).reshape(1, -1)

