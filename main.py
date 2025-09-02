"""Main Entry Point for AI Negotiator Showdown"""

import argparse
import logging
import random
from pathlib import Path
import json
import sys
from typing import List

# Import all components
from config import *
from agents.aggressive_trader import AggressiveTrader
from agents.smooth_diplomat import SmoothDiplomat
from agents.data_analyst import DataAnalyst
from agents.wildcard import Wildcard
from engine.tournament import Tournament
from ml_models.model_utils import ModelManager
from utils.logger import NegotiationLogger
from utils.visualizer import TournamentVisualizer

def setup_agents() -> List:
    """Create all agent instances"""
    agents = [
        AggressiveTrader("Aggressive_Alpha", "buyer"),
        AggressiveTrader("Aggressive_Beta", "seller"),
        SmoothDiplomat("Diplomat_Charlie", "buyer"),
        SmoothDiplomat("Diplomat_Delta", "seller"),
        DataAnalyst("Analyst_Echo", "buyer"),
        DataAnalyst("Analyst_Foxtrot", "seller"),
        Wildcard("Wildcard_Golf", "buyer"),
        Wildcard("Wildcard_Hotel", "seller")
    ]
    
    return agents


def train_models(model_manager: ModelManager, logger: NegotiationLogger):
    """Train all ML models"""
    print("\n🤖 Training ML Models...")
    print("=" * 50)
    
    # Generate training data
    datasets = model_manager.generate_training_data(num_negotiations=3000)
    print(f"✓ Generated {len(datasets['negotiation'])} negotiation records")
    print(f"✓ Generated {len(datasets['timeseries'])} time series points")
    
    # Train models
    training_results = model_manager.train_all_models(datasets)
    
    # Report results
    for model_name, result in training_results.items():
        if result["success"]:
            print(f"✓ {model_name.upper()} model trained successfully")
        else:
            print(f"✗ {model_name.upper()} training failed: {result.get('error', 'Unknown error')}")
    
    return training_results

def run_tournament(tournament_type: str, agents: List, model_manager: ModelManager, 
                  logger: NegotiationLogger, visualizer: TournamentVisualizer):
    """Run a complete tournament"""
    
    print(f"\n🏆 Starting {tournament_type.upper()} Tournament...")
    print("=" * 50)
    
    # Load models for agents
    model_loading = model_manager.load_models()
    trained_models = model_manager.get_models_for_agent()
    
    # Assign models to agents
    for agent in agents:
        agent.set_ml_models(trained_models)
    
    # Create and run tournament
    tournament = Tournament(agents, tournament_type)
    tournament.generate_matches()
    
    print(f"📋 Generated {len(tournament.matches)} matches")
    print("🔄 Running negotiations...")
    
    results = tournament.run_tournament()
    
    # Log results
    logger.log_tournament(results)
    
    # Display results
    print("\n🏅 TOURNAMENT RESULTS")
    print("=" * 50)
    
    if results["leaderboard"]:
        print(f"🥇 WINNER: {results['leaderboard'][0]['agent']} ({results['leaderboard'][0]['personality']})")
        print(f"   Score: {results['leaderboard'][0]['total_score']:.1f}")
        print(f"   Win Rate: {results['leaderboard'][0]['win_rate']:.1%}")
        print(f"   Avg Deal Time: {results['leaderboard'][0]['avg_deal_time']:.1f}s")
        
        print("\n📊 FULL LEADERBOARD:")
        for i, agent in enumerate(results["leaderboard"], 1):
            print(f"{i:2d}. {agent['agent']:20s} | {agent['total_score']:6.1f} pts | {agent['win_rate']:5.1%} wins | {agent['personality']}")
    
    print(f"\n📈 STATISTICS:")
    stats = results["statistics"]
    print(f"   Total Matches: {stats['total_matches']}")
    print(f"   Successful Deals: {stats['successful_deals']} ({stats['success_rate']:.1%})")
    print(f"   Average Deal Time: {stats.get('average_deal_time', 0):.1f}s")
    print(f"   Average Rounds: {stats.get('average_rounds', 0):.1f}")
    print(f"   Timeouts: {stats['timeouts']}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    visualizer.plot_leaderboard(results["leaderboard"], f"{LOG_DIR}/leaderboard_{tournament_type}.png")
    visualizer.plot_personality_analysis(results["detailed_results"], f"{LOG_DIR}/personality_{tournament_type}.png")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI Negotiator Showdown")
    parser.add_argument("--tournament", choices=["round_robin", "elimination", "grand_finals"], 
                       default="round_robin", help="Tournament type")
    parser.add_argument("--train", action="store_true", help="Train ML models first")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--num-negotiations", type=int, default=3000, 
                       help="Number of synthetic negotiations for training")
    
    args = parser.parse_args()
    
    # Setup
    logger = NegotiationLogger(str(LOG_DIR))
    model_manager = ModelManager(str(MODEL_DIR), str(DATA_DIR))
    visualizer = TournamentVisualizer()
    agents = setup_agents()
    
    print("🎯 AI NEGOTIATOR SHOWDOWN")
    print("=" * 50)
    print(f"🤝 Loaded {len(agents)} negotiation agents")
    print(f"🛒 Products: {', '.join(PRODUCTS.keys())}")
    print(f"⏱️  Time limit: {NEGOTIATION_TIME_LIMIT}s per negotiation")
    
    # Train models if requested
    training_results = None
    if args.train:
        training_results = train_models(model_manager, logger)
        
        if args.visualize and training_results:
            visualizer.plot_ml_model_performance(training_results, f"{LOG_DIR}/model_performance.png")
    
    # Run tournament
    tournament_results = run_tournament(args.tournament, agents, model_manager, logger, visualizer)
    
    print(f"\n✅ Tournament complete! Results saved to {LOG_DIR}")
    print("📁 Check the logs directory for detailed analysis and visualizations")

if __name__ == "__main__":
    main()

   
