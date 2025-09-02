"""Logging Configuration for AI Negotiator Showdown"""

import logging
import json
import csv
from datetime import datetime
from typing import Dict, List
import os
import matplotlib.pyplot as plt
import pandas as pd  # Added import




class NegotiationLogger:
    """Specialized logger for negotiation data"""
    
    def __init__(self, log_dir: str):
        """Initialize negotiation logger"""
        self.log_dir = log_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup main logger
        self.logger = logging.getLogger("ai_negotiator")
        self.logger.setLevel(logging.INFO)
        
        # File handler for general logs
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure directory exists
        file_handler = logging.FileHandler(f"{self.log_dir}/negotiator.log")
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(console_handler)
        
    def log_negotiation(self, result: Dict):
        """Log negotiation result to JSON file"""
        log_file = f"{self.log_dir}/negotiations.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    def log_tournament(self, tournament_result: Dict):
        """Log tournament results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f"{self.log_dir}/tournament_{timestamp}.json", "w") as f:
            json.dump(tournament_result, f, indent=2)
        
        # Save leaderboard CSV
        leaderboard_df = pd.DataFrame(tournament_result["leaderboard"])
        leaderboard_df.to_csv(f"{self.log_dir}/leaderboard_{timestamp}.csv", index=False)
        
    def get_historical_data(self) -> List[Dict]:
        """Load historical negotiation data"""
        log_file = f"{self.log_dir}/negotiations.jsonl"
        
        if not os.path.exists(log_file):
            return []
        
        negotiations = []
        with open(log_file, "r") as f:
            for line in f:
                try:
                    negotiations.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return negotiations

class TournamentVisualizer:
    """Visualizer for tournament results"""

    @staticmethod
    def plot_leaderboard(leaderboard, save_path=None, score_col="total_score"):
        import pandas as pd
        import matplotlib.pyplot as plt

        if isinstance(leaderboard, list):
            leaderboard = pd.DataFrame(leaderboard)

        if score_col not in leaderboard.columns:
            raise ValueError(
                f"Leaderboard missing '{score_col}' column. "
                f"Available: {leaderboard.columns.tolist()}"
            )

        plt.figure(figsize=(12, 6))
        plt.bar(leaderboard["agent"], leaderboard[score_col], color="skyblue")
        plt.xlabel("Agent")
        plt.ylabel(score_col.replace("_", " ").title())
        plt.title("Tournament Leaderboard")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def plot_personality_analysis(detailed_results, save_path=None, score_col="buyer_savings"):
        import pandas as pd
        import matplotlib.pyplot as plt

        if isinstance(detailed_results, list):
            detailed_results = pd.DataFrame(detailed_results)

        if "buyer_personality" not in detailed_results.columns or "seller_personality" not in detailed_results.columns:
            raise ValueError(
                f"'buyer_personality' or 'seller_personality' columns not found. "
                f"Available: {detailed_results.columns.tolist()}"
            )

    #   Flatten personalities into one column
        buyer_df = detailed_results[["buyer_personality", "buyer_savings"]].rename(
           columns={"buyer_personality": "personality", "buyer_savings": "score"}
        )
        seller_df = detailed_results[["seller_personality", "seller_profit"]].rename(
           columns={"seller_personality": "personality", "seller_profit": "score"}
        )

        combined = pd.concat([buyer_df, seller_df], ignore_index=True)

    # Group by personality
        grouped = combined.groupby("personality")["score"].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.bar(grouped["personality"], grouped["score"], color="lightcoral")
        plt.xlabel("Personality")
        plt.ylabel("Average Score (Savings/Profit)")
        plt.title("Personality Analysis (Buyers & Sellers)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
