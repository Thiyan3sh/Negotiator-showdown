# engine/game.py
"""Main Game Engine for Negotiations"""

from typing import Dict
import time
import logging
from datetime import datetime
from agents.base_agent import BaseAgent
from config import NEGOTIATION_TIME_LIMIT, MAX_ROUNDS_PER_NEGOTIATION

class NegotiationGame:
    """Manages a single negotiation session between buyer and seller"""
    
    def __init__(self, buyer: BaseAgent, seller: BaseAgent, product: str, 
                 product_info: Dict, buyer_budget: float, seller_min_price: float):
        """
        Initialize negotiation game
        
        Args:
            buyer: Buyer agent
            seller: Seller agent
            product: Product being negotiated
            product_info: Product information
            buyer_budget: Buyer's hidden budget constraint
            seller_min_price: Seller's hidden minimum price
        """
        self.buyer = buyer
        self.seller = seller
        self.product = product
        self.product_info = product_info
        self.start_time = None
        self.end_time = None
        self.rounds = []
        self.final_price = None
        self.success = False
        self.timeout = False
        
        # Set constraints and product info
        self.buyer.set_constraint(buyer_budget)
        self.seller.set_constraint(seller_min_price)
        self.buyer.set_product(product, product_info)
        self.seller.set_product(product, product_info)
        
        self.logger = logging.getLogger(__name__)
        
    def run_negotiation(self) -> Dict:
        """Run the complete negotiation process"""
        self.start_time = time.time()
        self.logger.info(f"Starting negotiation: {self.buyer.name} vs {self.seller.name} for {self.product}")
        
        try:
            current_round = 1
            current_agent = self.seller
            previous_offers = []
            
            while current_round <= MAX_ROUNDS_PER_NEGOTIATION:
                if time.time() - self.start_time > NEGOTIATION_TIME_LIMIT:
                    self.timeout = True
                    break
                
                round_start = time.time()
                
                if current_round == 1:
                    offer = current_agent.propose_offer(current_round, previous_offers)
                    response_type = "propose"
                    counter_offer = None
                    other_agent = self.buyer
                else:
                    last_offer = self.rounds[-1]["offer"]
                    response_type, counter_offer = current_agent.respond_offer(last_offer, current_round)
                    offer = counter_offer if counter_offer else last_offer
                    other_agent = self.seller if current_agent == self.buyer else self.buyer
                
                round_time = time.time() - round_start
                
                round_data = {
                    "round": current_round,
                    "agent": current_agent.name,
                    "agent_role": current_agent.role,
                    "response_type": response_type,
                    "offer": offer,
                    "time_taken": round_time,
                    "timestamp": datetime.now().isoformat()
                }
                self.rounds.append(round_data)
                self.logger.debug(f"Round {current_round}: {current_agent.name} {response_type} {offer}")
                
                if response_type == "accept":
                    self.success = True
                    self.final_price = offer
                    break
                elif response_type == "reject":
                    break
                elif response_type in ["counter", "propose"]:
                    previous_offers.append(offer)
                    
                current_agent = other_agent
                current_round += 1

        except Exception as e:
            self.logger.error(f"Error during negotiation: {e}")
            
        self.end_time = time.time()
        
        self.buyer.finalize_deal(self.final_price, self.success)
        self.seller.finalize_deal(self.final_price, self.success)
        
        return self._compile_results()
    
    def _compile_results(self) -> Dict:
        """Compile negotiation results"""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        results = {
            "game_id": f"{self.buyer.name}_{self.seller.name}_{int(self.start_time)}",
            "product": self.product,
            "buyer": self.buyer.name,
            "seller": self.seller.name,
            "buyer_personality": self.buyer.personality,
            "seller_personality": self.seller.personality,
            "buyer_constraint": self.buyer.constraint,
            "seller_constraint": self.seller.constraint,
            "success": self.success,
            "final_price": self.final_price,
            "rounds_completed": len(self.rounds),
            "duration_seconds": duration,
            "timeout": self.timeout,
            "rounds": self.rounds,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.success:
            results["buyer_savings"] = self.buyer.constraint - self.final_price
            results["seller_profit"] = self.final_price - self.seller.constraint
            
        return results
