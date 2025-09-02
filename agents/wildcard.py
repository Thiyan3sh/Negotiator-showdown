# agents/wildcard.py
"""Creative Wildcard Agent Implementation"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import random
from agents.base_agent import BaseAgent

class Wildcard(BaseAgent):
    """Unpredictable negotiation style with creative strategies"""
    
    def __init__(self, name: str, role: str):
        super().__init__(name, role, "wildcard")
        self.strategy_pool = ["anchoring", "reciprocity", "scarcity", "bundling", "time_pressure"]
        self.current_strategy = None
        self.strategy_state = {}
        
    def propose_offer(self, round_num: int, previous_offers: List[float]) -> float:
        """Propose creative offer based on dynamic strategy"""
        
        # Select strategy for this negotiation
        if round_num == 1:
            self.current_strategy = random.choice(self.strategy_pool)
            self.strategy_state = {"switches": 0, "last_switch": 0}
            
        # Switch strategy if stuck
        if round_num - self.strategy_state["last_switch"] > 3 and self.strategy_state["switches"] < 2:
            self.current_strategy = random.choice([s for s in self.strategy_pool if s != self.current_strategy])
            self.strategy_state["switches"] += 1
            self.strategy_state["last_switch"] = round_num
            
        # Execute strategy
        if self.current_strategy == "anchoring":
            offer = self._anchoring_strategy(round_num, previous_offers)
        elif self.current_strategy == "reciprocity":
            offer = self._reciprocity_strategy(round_num, previous_offers)
        elif self.current_strategy == "scarcity":
            offer = self._scarcity_strategy(round_num, previous_offers)
        elif self.current_strategy == "bundling":
            offer = self._bundling_strategy(round_num, previous_offers)
        else:  # time_pressure
            offer = self._time_pressure_strategy(round_num, previous_offers)
            
        return round(offer, 2)
    
    def respond_offer(self, offer: float, round_num: int) -> Tuple[str, Optional[float]]:
        """Respond creatively to offers"""
        utility = self.calculate_utility(offer)
        
        # Random acceptance for surprise factor
        if random.random() < 0.1 and utility > 0.3:
            return ("accept", None)
        
        # Strategy-specific responses
        if self.current_strategy == "reciprocity":
            # Mirror opponent's concession pattern
            if hasattr(self, "_last_opponent_offer"):
                concession = abs(offer - self._last_opponent_offer)
                if concession > 0.1:  # Significant concession
                    return ("accept", None) if utility > 0.4 else ("counter", self._reciprocate(offer))
            self._last_opponent_offer = offer
            
        elif self.current_strategy == "time_pressure":
            # Increase pressure near deadline
            if round_num > 7:
                if utility > 0.2:
                    return ("accept", None)
                else:
                    # Final offer
                    return ("counter", self.constraint)
                    
        # Default creative response
        if utility > 0.6:
            return ("accept", None)
        elif not self.is_acceptable(offer):
            # Surprise counter
            counter = self._surprise_counter(offer, round_num)
            return ("counter", counter)
        else:
            # Pattern-breaking counter
            counter = self._pattern_break(offer, round_num)
            return ("counter", counter)
    
    def _anchoring_strategy(self, round_num: int, previous_offers: List[float]) -> float:
        """Extreme initial anchor, then gradual concessions"""
        if round_num == 1:
            # Set extreme anchor
            if self.role == "buyer":
                return self.product_info["min_price"] * 0.5
            else:
                return self.product_info["max_price"] * 2
        else:
            # Slow concessions from anchor
            if "anchor" not in self.strategy_state:
                self.strategy_state["anchor"] = previous_offers[0] if previous_offers else self.constraint
                
            anchor = self.strategy_state["anchor"]
            concession_rate = 0.15 * round_num
            
            if self.role == "buyer":
                offer = min(anchor * (1 + concession_rate), self.constraint)
            else:
                offer = max(anchor * (1 - concession_rate * 0.5), self.constraint)
            return offer
    
    def _reciprocity_strategy(self, round_num: int, previous_offers: List[float]) -> float:
        """Match opponent's concession patterns"""
        if round_num == 1 or not previous_offers:
            # Start moderately
            price_range = self.product_info["max_price"] - self.product_info["min_price"]
            if self.role == "buyer":
                return self.product_info["min_price"] + price_range * 0.35
            else:
                return self.product_info["min_price"] + price_range * 0.65
        else:
            # Calculate opponent's average concession
            if len(previous_offers) > 1:
                concessions = [abs(previous_offers[i] - previous_offers[i-1]) 
                             for i in range(1, len(previous_offers))]
                avg_concession = np.mean(concessions) if concessions else 0.05
            else:
                avg_concession = 0.1
                
            # Match with slight advantage
            last_offer = previous_offers[-1]
            if self.role == "buyer":
                offer = min(last_offer + avg_concession * 0.8, self.constraint)
            else:
                offer = max(last_offer - avg_concession * 0.8, self.constraint)
            return offer
    
    def _scarcity_strategy(self, round_num: int, previous_offers: List[float]) -> float:
        """Create urgency through artificial scarcity"""
        # Pretend limited availability
        scarcity_factor = 1 + (0.1 * round_num)  # Increase price pressure
        
        if round_num == 1:
            if self.role == "buyer":
                # Act uninterested initially
                return self.product_info["min_price"] * 0.7
            else:
                # Premium pricing for "limited stock"
                return self.product_info["max_price"] * 1.2
        else:
            # Gradually reveal "urgency"
            if self.role == "buyer":
                # Sudden increase in interest
                base = previous_offers[-1] if previous_offers else self.constraint
                offer = min(base * scarcity_factor, self.constraint)
            else:
                # Decrease price as "stock runs out"
                base = previous_offers[-1] if previous_offers else self.constraint
                offer = max(base * (1 - 0.05 * round_num), self.constraint)
            return offer
    
    def _bundling_strategy(self, round_num: int, previous_offers: List[float]) -> float:
        """Simulate bundle pricing adjustments"""
        # Pretend to adjust for bulk or package deals
        bundle_modifier = 1 + 0.05 * random.choice([-1, 1])  # ±5% random adjustment
        
        if round_num == 1:
            base_price = (self.product_info["min_price"] + self.product_info["max_price"]) / 2
            offer = base_price * bundle_modifier
        else:
            last_offer = previous_offers[-1] if previous_offers else self.constraint
            offer = last_offer * bundle_modifier
            
        # Ensure within constraint
        if self.role == "buyer":
            return min(offer, self.constraint)
        else:
            return max(offer, self.constraint)
    
    def _time_pressure_strategy(self, round_num: int, previous_offers: List[float]) -> float:
        """Increase concessions under time pressure"""
        pressure_factor = round_num / 10  # Increase pressure over time
        
        if round_num == 1:

# Start conservatively
            if self.role == "buyer":
                return self.constraint * 0.6
            else:
                return self.constraint * 1.4
        else:
            # Accelerate concessions
            concession = 0.1 * pressure_factor
            last_offer = previous_offers[-1] if previous_offers else self.constraint
            
            if self.role == "buyer":
                offer = min(last_offer * (1 + concession), self.constraint)
            else:
                offer = max(last_offer * (1 - concession), self.constraint)
            return offer
    
    def _surprise_counter(self, offer: float, round_num: int) -> float:
        """Generate unexpected counter-offer"""
        # Random dramatic move
        if random.random() < 0.5:
            # Big jump
            if self.role == "buyer":
                return min(offer * 1.5, self.constraint)
            else:
                return max(offer * 0.7, self.constraint)
        else:
            # Tiny adjustment
            if self.role == "buyer":
                return min(offer * 1.02, self.constraint)
            else:
                return max(offer * 0.98, self.constraint)
    
    def _pattern_break(self, offer: float, round_num: int) -> float:
        """Break negotiation patterns"""
        # Intentionally non-linear response
        noise = random.uniform(-0.1, 0.1)
        if self.role == "buyer":
            return min(offer * (1 + 0.1 + noise), self.constraint)
        else:
            return max(offer * (1 - 0.1 + noise), self.constraint)
    
    def _reciprocate(self, offer: float) -> float:
        """Reciprocate opponent's concession"""
        if hasattr(self, "_last_opponent_offer"):
            concession = abs(offer - self._last_opponent_offer)
            if self.role == "buyer":
                return min(offer + concession, self.constraint)
            else:
                return max(offer - concession, self.constraint)
        return offer
