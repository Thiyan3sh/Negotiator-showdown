# engine/tournament.py
"""Tournament Management System"""

import itertools
import random
from typing import List, Dict, Optional
import logging
from engine.game import NegotiationGame
from engine.scoring import ScoreCalculator
from agents.base_agent import BaseAgent
from config import PRODUCTS, TOURNAMENT_CONFIG

class Tournament:
    """Manages different tournament formats"""
    
    def __init__(self, agents: List[BaseAgent], tournament_type: str = "round_robin"):
        """
        Initialize tournament
        
        Args:
            agents: List of participating agents
            tournament_type: Type of tournament (round_robin, elimination, grand_finals)
        """
        self.agents = agents
        self.tournament_type = tournament_type
        self.matches = []
        self.results = []
        self.leaderboard = []
        self.scorer = ScoreCalculator()
        self.logger = logging.getLogger(__name__)
        
    def generate_matches(self) -> List[Dict]:
        """Generate match schedule based on tournament type"""
        matches = []
        
        if self.tournament_type == "round_robin":
            # Each agent plays each other as both buyer and seller
            for agent1, agent2 in itertools.combinations(self.agents, 2):
                for product in PRODUCTS.keys():
                    # Generate realistic constraints
                    product_info = PRODUCTS[product]
                    base_price = (product_info["min_price"] + product_info["max_price"]) / 2
                    
                    # Match 1: agent1 as buyer, agent2 as seller
                    buyer_budget = base_price * random.uniform(0.8, 1.5)
                    seller_min = base_price * random.uniform(0.5, 1.2)
                    
                    matches.append({
                        "id": f"rr_{agent1.name}_{agent2.name}_{product}_1",
                        "buyer": agent1,
                        "seller": agent2,
                        "product": product,
                        "buyer_budget": buyer_budget,
                        "seller_min_price": seller_min
                    })
                    
                    # Match 2: agent2 as buyer, agent1 as seller
                    buyer_budget = base_price * random.uniform(0.8, 1.5)
                    seller_min = base_price * random.uniform(0.5, 1.2)
                    
                    matches.append({
                        "id": f"rr_{agent2.name}_{agent1.name}_{product}_2",
                        "buyer": agent2,
                        "seller": agent1,
                        "product": product,
                        "buyer_budget": buyer_budget,
                        "seller_min_price": seller_min
                    })
                    
        elif self.tournament_type == "elimination":
            # Single elimination bracket
            random.shuffle(self.agents)
            round_agents = self.agents.copy()
            
            while len(round_agents) > 1:
                next_round = []
                for i in range(0, len(round_agents), 2):
                    if i + 1 < len(round_agents):
                        agent1, agent2 = round_agents[i], round_agents[i + 1]
                        winner = self._head_to_head_match(agent1, agent2)
                        next_round.append(winner)
                        
                        # Record elimination match
                        product = random.choice(list(PRODUCTS.keys()))
                        product_info = PRODUCTS[product]
                        base_price = (product_info["min_price"] + product_info["max_price"]) / 2
                        
                        matches.append({
                            "id": f"elim_{agent1.name}_{agent2.name}_{product}",
                            "buyer": agent1,
                            "seller": agent2,
                            "product": product,
                            "buyer_budget": base_price * random.uniform(0.8, 1.5),
                            "seller_min_price": base_price * random.uniform(0.5, 1.2),
                            "elimination": True
                        })
                    else:
                        # Bye
                        next_round.append(round_agents[i])
                        
                round_agents = next_round
                
        self.matches = matches
        return matches
    
    def run_tournament(self) -> Dict:
        """
        Execute the complete tournament
        
        Returns:
            Tournament results and statistics
        """
        self.logger.info(f"Starting {self.tournament_type} tournament with {len(self.agents)} agents")
        
        # Generate matches if not already done
        if not self.matches:
            self.generate_matches()
        
        # Run all matches
        for match in self.matches:
            result = self._run_match(match)
            self.results.append(result)
            
        # Calculate final rankings
        self.leaderboard = self._calculate_rankings()
        
        tournament_results = {
            "tournament_type": self.tournament_type,
            "participants": [agent.name for agent in self.agents],
            "total_matches": len(self.matches),
            "completed_matches": len(self.results),
            "leaderboard": self.leaderboard,
            "detailed_results": self.results,
            "statistics": self._calculate_statistics()
        }
        
        self.logger.info(f"Tournament completed. Winner: {self.leaderboard[0]['agent'] if self.leaderboard else 'None'}")
        
        return tournament_results
    
    def _run_match(self, match: Dict) -> Dict:
        """Run a single match"""
        game = NegotiationGame(
            buyer=match["buyer"],
            seller=match["seller"],
            product=match["product"],
            product_info=PRODUCTS[match["product"]],
            buyer_budget=match["buyer_budget"],
            seller_min_price=match["seller_min_price"]
        )
        
        result = game.run_negotiation()
        result["match_id"] = match["id"]
        
        return result
    
    def _head_to_head_match(self, agent1: BaseAgent, agent2: BaseAgent) -> BaseAgent:
        """Determine winner between two agents"""
        # Run multiple matches and compare total scores
        scores = {agent1.name: 0, agent2.name: 0}
        
        for _ in range(TOURNAMENT_CONFIG["elimination"]["best_of"]):
            product = random.choice(list(PRODUCTS.keys()))
            product_info = PRODUCTS[product]
            base_price = (product_info["min_price"] + product_info["max_price"]) / 2
            
            # Match 1: agent1 buyer, agent2 seller
            game1 = NegotiationGame(
                buyer=agent1, seller=agent2, product=product, product_info=product_info,
                buyer_budget=base_price * random.uniform(0.8, 1.5),
                seller_min_price=base_price * random.uniform(0.5, 1.2)
            )
            result1 = game1.run_negotiation()
            
            # Match 2: agent2 buyer, agent1 seller
            game2 = NegotiationGame(
                buyer=agent2, seller=agent1, product=product, product_info=product_info,
                buyer_budget=base_price * random.uniform(0.8, 1.5),
                seller_min_price=base_price * random.uniform(0.5, 1.2)
            )
            result2 = game2.run_negotiation()
            
            # Calculate scores
            if result1["success"]:
                score1 = self.scorer.calculate_agent_score(result1, agent1.name)
                score2 = self.scorer.calculate_agent_score(result1, agent2.name)
                scores[agent1.name] += score1
                scores[agent2.name] += score2
                
            if result2["success"]:
                score1 = self.scorer.calculate_agent_score(result2, agent1.name)
                score2 = self.scorer.calculate_agent_score(result2, agent2.name)
                scores[agent1.name] += score1
                scores[agent2.name] += score2
        
        return agent1 if scores[agent1.name] >= scores[agent2.name] else agent2
    
    def _calculate_rankings(self) -> List[Dict]:
        """Calculate final tournament rankings"""
        agent_stats = {}
        
        # Initialize stats for all agents
        for agent in self.agents:
            agent_stats[agent.name] = {
                "agent": agent.name,
                "personality": agent.personality,
                "total_score": 0,
                "matches_played": 0,
                "deals_made": 0,
                "avg_deal_time": 0,
                "total_profit_savings": 0,
                "win_rate": 0
            }
        
        # Process results
        total_times = {agent.name: [] for agent in self.agents}
        
        for result in self.results:
            buyer_name = result["buyer"]
            seller_name = result["seller"]
            
            # Update match counts
            agent_stats[buyer_name]["matches_played"] += 1
            agent_stats[seller_name]["matches_played"] += 1
            
            if result["success"]:
                # Update deal counts
                agent_stats[buyer_name]["deals_made"] += 1
                agent_stats[seller_name]["deals_made"] += 1
                
                # Calculate scores
                buyer_score = self.scorer.calculate_agent_score(result, buyer_name)
                seller_score = self.scorer.calculate_agent_score(result, seller_name)
                
                agent_stats[buyer_name]["total_score"] += buyer_score
                agent_stats[seller_name]["total_score"] += seller_score
                
                # Update profit/savings
                if "buyer_savings" in result:
                    agent_stats[buyer_name]["total_profit_savings"] += result["buyer_savings"]
                if "seller_profit" in result:
                    agent_stats[seller_name]["total_profit_savings"] += result["seller_profit"]
                
                # Track times
                total_times[buyer_name].append(result["duration_seconds"])
                total_times[seller_name].append(result["duration_seconds"])
        
        # Calculate averages and win rates
        for agent_name, stats in agent_stats.items():
            if stats["matches_played"] > 0:
                stats["win_rate"] = stats["deals_made"] / stats["matches_played"]
                stats["avg_score"] = stats["total_score"] / stats["matches_played"]
                
            if total_times[agent_name]:
                stats["avg_deal_time"] = sum(total_times[agent_name]) / len(total_times[agent_name])
        
        # Sort by total score
        leaderboard = sorted(agent_stats.values(), key=lambda x: x["total_score"], reverse=True)
        
        return leaderboard
    
    def _calculate_statistics(self) -> Dict:
        """Calculate tournament statistics"""
        successful_deals = [r for r in self.results if r["success"]]
        
        stats = {
            "total_matches": len(self.results),
            "successful_deals": len(successful_deals),
            "success_rate": len(successful_deals) / len(self.results) if self.results else 0,
            "average_deal_time": 0,
            "average_rounds": 0,
            "timeouts": len([r for r in self.results if r.get("timeout", False)])
        }
        
        if successful_deals:
            stats["average_deal_time"] = sum(r["duration_seconds"] for r in successful_deals) / len(successful_deals)
            stats["average_rounds"] = sum(r["rounds_completed"] for r in successful_deals) / len(successful_deals)
            stats["average_final_price"] = sum(r["final_price"] for r in successful_deals) / len(successful_deals)
        
        return stats


