"""Base Agent class for negotiation"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import random
import logging
from datetime import datetime
import numpy as np

class BaseAgent(ABC):
    """Abstract base class for negotiation agents"""
    
    def __init__(self, name: str, role: str, personality: str):
        """
        Initialize base agent
        
        Args:
            name: Agent identifier
            role: 'buyer' or 'seller'
            personality: Agent personality type
        """
        self.name = name
        self.role = role
        self.personality = personality
        self.negotiation_history = []
        self.current_product = None
        self.constraint = None  # Hidden constraint (max budget for buyer, min price for seller)
        self.ml_models = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    def set_constraint(self, constraint: float):
        """Set hidden constraint for the agent"""
        self.constraint = constraint
        
    def set_product(self, product: str, product_info: Dict):
        """Set current product being negotiated"""
        self.current_product = product
        self.product_info = product_info
        
    def set_ml_models(self, models: Dict):
        """Attach ML models for decision making"""
        self.ml_models = models
        
    @abstractmethod
    def propose_offer(self, round_num: int, previous_offers: List[float]) -> float:
	    
      """
        Propose an offer based on strategy
        
        Args:
            round_num: Current round number
            previous_offers: History of previous offers
            
        Returns:
            Proposed offer price
        """
    pass
    
    @abstractmethod
    def respond_offer(self, offer: float, round_num: int) -> Tuple[str, Optional[float]]:
        """
        Respond to an offer
        
        Args:
            offer: Received offer
            round_num: Current round number
            
        Returns:
            Tuple of (response_type, counter_offer)
            response_type: 'accept', 'reject', 'counter'
        """
    pass
    
    def finalize_deal(self, final_price: Optional[float], success: bool):
        """
        Finalize negotiation and record results
        
        Args:
            final_price: Agreed price if successful
            success: Whether deal was successful
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "product": self.current_product,
            "role": self.role,
            "final_price": final_price,
            "success": success,
            "constraint": self.constraint
        }
        self.negotiation_history.append(result)
        self.logger.info(f"Deal finalized: {result}")
        
    def calculate_utility(self, price: float) -> float:
        """Calculate utility of a given price"""
        if self.role == "buyer":
            # Lower price is better for buyer
            if price > self.constraint:
                return -1  # Infeasible
            return (self.constraint - price) / self.constraint
        else:  # seller
            # Higher price is better for seller
            if price < self.constraint:
                return -1  # Infeasible
            max_expected = self.product_info["max_price"]
            return (price - self.constraint) / (max_expected - self.constraint)
    
    def is_acceptable(self, price: float) -> bool:
        """Check if price meets constraint"""
        if self.role == "buyer":
            return price <= self.constraint
        else:
            return price >= self.constraint
    
    def get_personality_traits(self) -> Dict:
        """Get personality-specific traits"""
        from config import PERSONALITY_PARAMS
        return PERSONALITY_PARAMS.get(self.personality.lower(), PERSONALITY_PARAMS["diplomatic"])
