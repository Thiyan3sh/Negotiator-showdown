"""Scoring System for Negotiation Performance"""

import numpy as np
from typing import Dict
from config import SCORING_WEIGHTS, PERSONALITY_PARAMS

class ScoreCalculator:
    """Calculates scores for negotiation performance"""
    
    def __init__(self):
        self.weights = SCORING_WEIGHTS

    def calculate_agent_score(self, negotiation_result: Dict, agent_name: str) -> float:
        """
        Calculate comprehensive score for an agent's performance
        
        Args:
            negotiation_result: Complete negotiation result
            agent_name: Name of agent to score
            
        Returns:
            Total weighted score (0-100)
        """
        if not negotiation_result.get("success", False):
            return 0.0
        
        # Determine agent role
        agent_role = "buyer" if negotiation_result.get("buyer") == agent_name else "seller"
        
        # Calculate component scores
        profit_score = self._calculate_profit_score(negotiation_result, agent_role)
        consistency_score = self._calculate_consistency_score(negotiation_result, agent_name)
        speed_score = self._calculate_speed_score(negotiation_result)
        
        # Weighted final score
        total_score = (
            profit_score * self.weights.get("profit_savings", 0.5) +
            consistency_score * self.weights.get("character_consistency", 0.3) +
            speed_score * self.weights.get("speed_bonus", 0.2)
        ) * 100
        
        return round(total_score, 2)
    
    def _calculate_profit_score(self, result: Dict, agent_role: str) -> float:
        """Calculate profit/savings score (0-1)"""
        buyer_constraint = result.get("buyer_constraint", 0)
        seller_constraint = result.get("seller_constraint", 0)
        
        if agent_role == "buyer":
            savings = result.get("buyer_savings", 0)
            max_possible_savings = buyer_constraint - seller_constraint
            if max_possible_savings <= 0:
                return 0.5  # Neutral score if no room
            return min(1.0, savings / max_possible_savings)
        else:  # seller
            profit = result.get("seller_profit", 0)
            max_possible_profit = buyer_constraint - seller_constraint
            if max_possible_profit <= 0:
                return 0.5
            return min(1.0, profit / max_possible_profit)
    
    def _calculate_consistency_score(self, result: Dict, agent_name: str) -> float:
        """Calculate personality consistency score (0-1)"""
        agent_role = "buyer" if result.get("buyer") == agent_name else "seller"
        personality = result.get(f"{agent_role}_personality", "unknown")
        rounds = result.get("rounds", [])

        # Use safe defaults if personality not found
        expected_traits = PERSONALITY_PARAMS.get(personality.lower(), {
            "concession_rate": 0.1,
            "patience": 0.5
        })

        consistency_factors = []

        # Filter agent rounds
        agent_rounds = [r for r in rounds if r.get("agent") == agent_name]
        if len(agent_rounds) > 1:
            offers = [r["offer"] for r in agent_rounds if r.get("response_type") in ["propose", "counter"]]
            if len(offers) > 1:
                concessions = []
                for i in range(1, len(offers)):
                    if agent_role == "buyer":
                        concession = (offers[i] - offers[i-1]) / offers[i-1] if offers[i-1] != 0 else 0
                    else:
                        concession = (offers[i-1] - offers[i]) / offers[i-1] if offers[i-1] != 0 else 0
                    concessions.append(abs(concession))
                avg_concession = np.mean(concessions) if concessions else 0
                expected_concession = expected_traits.get("concession_rate", 0.1)
                concession_consistency = 1 - min(1.0, abs(avg_concession - expected_concession) / expected_concession)
                consistency_factors.append(concession_consistency)

        # Analyze response times (patience)
        response_times = [r.get("time_taken", 0) for r in agent_rounds]
        if response_times:
            avg_response_time = np.mean(response_times)
            normalized_time = min(1.0, avg_response_time / 30)  # Assume max 30 sec per response
            expected_patience = expected_traits.get("patience", 0.5)
            patience_consistency = 1 - abs(normalized_time - expected_patience)
            consistency_factors.append(patience_consistency)

        return np.mean(consistency_factors) if consistency_factors else 0.5
    
    def _calculate_speed_score(self, result: Dict) -> float:
        """Calculate speed bonus score (0-1)"""
        duration = result.get("duration_seconds", 0)
        max_time = 180  # 3 minutes
        if duration <= 0:
            return 0.0
        speed_score = max(0.0, 1 - (duration / max_time))
        return speed_score ** 0.5
    
    def calculate_tournament_mvp(self, tournament_results: Dict) -> Dict:
        """Calculate Most Valuable Player of tournament"""
        leaderboard = tournament_results.get("leaderboard", [])
        if not leaderboard:
            return None
        mvp = leaderboard[0]
        return {
            "name": mvp.get("agent", "unknown"),
            "personality": mvp.get("personality", "unknown"),
            "total_score": mvp.get("total_score", 0),
            "win_rate": mvp.get("win_rate", 0),
            "avg_deal_time": mvp.get("avg_deal_time", 0),
            "deals_completed": mvp.get("deals_made", 0),
            "total_earnings": mvp.get("total_profit_savings", 0)
        }
