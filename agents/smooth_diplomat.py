"""Smooth Diplomat Agent Implementation"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from agents.base_agent import BaseAgent

class SmoothDiplomat(BaseAgent):
    """Diplomatic negotiation style - fair offers, gradual concessions, win-win focus"""
    
    def __init__(self, name: str, role: str):
        super().__init__(name, role, "diplomatic")
        self.fairness_weight = 0.6
        self.relationship_value = 0.3
        
    def propose_offer(self, round_num: int, previous_offers: List[float]) -> float:
        """Propose diplomatic offer"""
        traits = self.get_personality_traits()
        
        if round_num == 1:
            # Start with reasonable offer
            price_range = self.product_info["max_price"] - self.product_info["min_price"]
            if self.role == "buyer":
                # Offer slightly below middle
                offer = self.product_info["min_price"] + price_range * 0.4
            else:
                # Ask slightly above middle
                offer = self.product_info["min_price"] + price_range * 0.6
        else:
            # Gradual concessions based on pattern
            if previous_offers:
                # Analyze opponent's concession pattern
                if len(previous_offers) > 1:
                    opponent_concession = abs(previous_offers[-1] - previous_offers[-2])
                    my_concession = opponent_concession * (1 + self.fairness_weight)
                else:
                    my_concession = traits["concession_rate"]
                
                last_offer = previous_offers[-1]
                
                # Use LSTM for sequence prediction if available
                if "lstm" in self.ml_models and self.ml_models["lstm"]:
                    sequence = np.array(previous_offers[-5:] if len(previous_offers) >= 5 else previous_offers)
                    next_prediction = self.ml_models["lstm"].predict(sequence.reshape(1, -1, 1))
                    target = next_prediction[0][0]
                    
                    # Move towards predicted acceptable price
                    if self.role == "buyer":
                        offer = min(last_offer + my_concession * price_range, target, self.constraint)
                    else:
                        offer = max(last_offer - my_concession * price_range, target, self.constraint)
                else:
                    if self.role == "buyer":
                        offer = min(last_offer * (1 + my_concession), self.constraint)
                    else:
                        offer = max(last_offer * (1 - my_concession), self.constraint)
            else:
                # No previous offers, use default
                offer = self.constraint * (0.9 if self.role == "seller" else 1.1)
                
        return round(offer, 2)
    
    def respond_offer(self, offer: float, round_num: int) -> Tuple[str, Optional[float]]:
        """Respond diplomatically to offers"""
        utility = self.calculate_utility(offer)
        
        # Accept reasonable offers
        if utility > 0.5:
            return ("accept", None)
        
        # Never reject outright unless impossible
        if not self.is_acceptable(offer):
            # Propose meeting in the middle
            if self.role == "buyer":
                counter = min((offer + self.constraint) / 2, self.constraint)
            else:
                counter = max((offer + self.constraint) / 2, self.constraint)
            return ("counter", round(counter, 2))
        
        # Calculate fair counter-offer
        if utility > 0.3:
            # Close to acceptable, small adjustment
            adjustment = 0.05 if self.role == "buyer" else -0.05
            counter = offer * (1 + adjustment)
        else:
            # Need bigger move
            counter = self.propose_offer(round_num, [offer])
            
        return ("counter", round(counter, 2))
    
    def _calculate_fairness_score(self, price: float) -> float:
	
        """Calculate how fair a price is"""
        price_range = self.product_info["max_price"] - self.product_info["min_price"]
        middle = self.product_info["min_price"] + price_range / 2
        deviation = abs(price - middle) / price_range
        return 1 - deviation