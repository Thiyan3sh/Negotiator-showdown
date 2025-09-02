"""Logging Configuration for AI Negotiator Showdown"""
import pandas as pd
import logging
import json
import csv
from datetime import datetime
from typing import Dict, List
import os

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

