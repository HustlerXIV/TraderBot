# Forex AI Trading Bot - Quick Start Guide

## What This Bot Does
An AI-powered Forex trading bot that uses **4 machine learning models** working together (ensemble) to predict currency price movements and trade automatically.

Supports **OANDA** (Mac/Linux/Windows) and **MetaTrader 5** (Windows).

### AI Models Used
| Model | Strength |
|-------|----------|
| Random Forest | Captures non-linear patterns in price data |
| XGBoost | Best-in-class gradient boosting with regularization |
| Gradient Boosting | Adds diversity to the ensemble |
| LSTM (Deep Learning) | Learns sequential patterns in time series |

The bot combines all models' predictions using weighted voting, where weights are automatically optimized based on each model's performance.

---

## Setup for Mac (OANDA) - 5 Minutes

### Step 1: Create Virtual Environment & Install
```bash
cd "Trader Bot"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Get Your Free OANDA Practice Account

1. Go to **https://www.oanda.com/** and click "Try free demo"
2. Sign up for a free practice account (no credit card needed)
3. Once logged in, go to **Account Settings > API Access > Personal Access Token**
4. Click **"Generate"** to create your API token - copy it somewhere safe
5. Note your **Account ID** - visible on your account page (format: `101-001-12345678-001`)

### Step 3: Run Backtest First (Recommended!)
```bash
python3 -m forex_ai_bot --backtest --symbol EURUSD --bars 3000
```
This tests the AI on simulated historical data - no OANDA account needed.

### Step 4: Start Paper Trading with OANDA (No Real Money)
```bash
python3 -m forex_ai_bot --token YOUR_API_TOKEN --account YOUR_ACCOUNT_ID
```
The bot starts in **paper trading mode** by default.

### Step 5: Go Live (When Ready)
```bash
python3 -m forex_ai_bot --live --no-practice --token YOUR_API_TOKEN --account YOUR_ACCOUNT_ID
```

**Or** save credentials in `config.py` so you don't need to type them every time:
```python
@dataclass
class OandaConfig:
    api_token: str = "your-api-token-here"
    account_id: str = "101-001-12345678-001"
    practice: bool = True       # True = demo, False = live
```

---

## Setup for Windows (MT5) - Alternative

If you prefer MetaTrader 5 (Windows only):
```bash
pip install MetaTrader5
python3 -m forex_ai_bot --broker mt5 --login 12345678 --password "pass" --server "Broker-Server"
```

---

## How the AI Learns

1. **Initial Training**: On first run, the bot downloads historical data and trains all models
2. **Continuous Learning**: Every 24 hours (configurable), the bot:
   - Evaluates its recent trading performance
   - Collects fresh market data
   - Retrains models with new data
   - Validates the new model before deploying it
   - Automatically adjusts ensemble weights based on which models perform best
3. **Safety Check**: If a retrained model performs worse than random chance, the bot automatically rolls back to the previous model

---

## Risk Management Built In

- Max 1% risk per trade (configurable)
- Max 5 simultaneous trades
- ATR-based stop loss & take profit
- Trailing stops
- Max 10% drawdown circuit breaker (auto-stops trading)
- Consecutive loss cooldown (pauses after 5 losses in a row)

---

## Commands

| Command | What it does |
|---------|-------------|
| `python3 -m forex_ai_bot` | Start paper trading |
| `python3 -m forex_ai_bot --backtest` | Backtest on historical data |
| `python3 -m forex_ai_bot --backtest --symbol GBPUSD` | Backtest specific pair |
| `python3 -m forex_ai_bot --train` | Train/retrain models only |
| `python3 -m forex_ai_bot --status` | Show bot & model status |
| `python3 -m forex_ai_bot --token X --account Y` | Start with OANDA credentials |
| `python3 -m forex_ai_bot --live` | LIVE trading (real money!) |
| `python3 -m forex_ai_bot --broker mt5` | Use MetaTrader 5 instead |

---

## Project Structure
```
forex_ai_bot/
├── config.py            # All settings (edit this!)
├── bot.py               # Main orchestrator
├── oanda_connector.py   # OANDA REST API (Mac/Linux/Windows)
├── mt5_connector.py     # MetaTrader 5 (Windows only)
├── feature_engine.py    # Technical indicators (80+ features)
├── ensemble_model.py    # 4 ML models + ensemble logic
├── risk_manager.py      # Risk management system
├── strategy.py          # Signal generation + filters
├── execution_engine.py  # Order execution + management
├── continuous_learner.py # Auto-retraining pipeline
├── data/                # Historical data cache
├── models/              # Saved ML models
├── logs/                # Trading logs
└── checkpoints/         # Model checkpoints (rollback)
```

---

## Important Warnings

- **Always start with paper trading** to validate the bot's performance
- **Past performance does not guarantee future results** - this is true for all trading systems
- **Never risk money you can't afford to lose**
- The bot's AI models need time to learn - give it at least a few days of paper trading
- Monitor the logs regularly, especially in the first few weeks
