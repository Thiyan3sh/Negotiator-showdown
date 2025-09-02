# 🤝 Negotiator Showdown

An AI-powered **multi-agent negotiation simulator** where agents with different personalities compete in tournaments.  
Agents negotiate over multiple products, make offers, concessions, and adapt strategies.  
Results are logged, analyzed, and visualized for transparency.

---

## 🚀 Features

- **Tournament Engine**
  - Supports Round-Robin, Elimination, and Grand Finals.
  - Generates scores, leaderboards, and statistics automatically.

- **Agent Personalities**
  - Aggressive, Diplomatic, Analytical, and Wildcard.
  - Each personality follows unique concession and negotiation logic.
  - Personality impacts win rates and deal success.

- **Negotiation Logging**
  - Every offer, counter-offer, concession, and outcome is stored.
  - Enables **step-by-step replay** of negotiations.

- **Machine Learning Integration**
  - ML models (XGBoost, LSTM, SARIMA) can be trained to predict offers and outcomes.
  - Flexible design — new models can be added easily.

- **Visualization**
  - Leaderboard charts, personality heatmaps, and performance analysis.
  - Results stored in `logs/` folder for later review.

---

## 📂 Project Structure

```
negotiator_showdown/
│── agents/                # Agent personality definitions
│   ├── aggressive.py
│   ├── diplomatic.py
│   ├── analytical.py
│   └── wildcard.py
│
│── ml_models/              # Machine Learning models
│   ├── saved_models/       # Trained models (XGBoost, LSTM, SARIMA)
│   └── model_manager.py
│
│── core/                   # Core tournament logic
│   ├── tournament.py
│   ├── negotiation.py
│   └── utils.py
│
│── logs/                   # Logs and visualizations
│
│── main.py                 # Entry point (run tournament)
│── requirements.txt
│── README.md
```

---

## ⚡ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/negotiator_showdown.git
   cd negotiator_showdown
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Train ML models:
   ```bash
   python ml_models/model_manager.py --train
   ```

---

## ▶️ Usage

Run a tournament:

```bash
python main.py --tournament round_robin
```

Other options:
```bash
python main.py --tournament elimination
python main.py --tournament grand_final
```

---

## 📊 Example Output

```
📋 Generated 280 matches
🔄 Running negotiations...

🏅 TOURNAMENT RESULTS
==================================================
🥇 WINNER: Aggressive_Beta (aggressive)
   Score: 1380.3
   Win Rate: 60.0%
   Avg Deal Time: 0.0s

📊 FULL LEADERBOARD:
 1. Aggressive_Beta      | 1380.3 pts | 60.0% wins | aggressive
 2. Wildcard_Hotel       | 1254.7 pts | 44.3% wins | wildcard
 ...

📈 STATISTICS:
   Total Matches: 280
   Successful Deals: 111 (39.6%)
   Average Rounds: 4.3
   Timeouts: 0

✅ Tournament complete! Results saved to logs/
```

- Negotiation logs available in `logs/`.
- Visualizations include leaderboards, performance graphs, and personality comparisons.

---

## 👥 Team Contributions

- **Tournament Engine & Match Simulation** – Designed round-robin, elimination, and finals logic.
- **Agent Personalities & Negotiation Logic** – Implemented strategies for Aggressive, Diplomatic, Analytical, and Wildcard agents.
- **ML Model Training & Prediction** – Integrated XGBoost, LSTM, SARIMA for strategy improvement.
- **Visualization & Logging System** – Built charts, logging, and replay system for negotiations.

---

## 🌟 Unique Highlights

- **Transparent Negotiations:** Not just scores — every step of the deal-making process is logged.
- **Human-like Agents:** Personality-driven strategies simulate real-world negotiators.
- **Extensible Design:** Add new agents, ML models, or tournament formats easily.
- **Research-Oriented:** Useful for behavioral economics, AI competitions, and training simulators.

---

## 📜 License

This project is for educational and research purposes.  
Modify and extend freely for non-commercial use.
