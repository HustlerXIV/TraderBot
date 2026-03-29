"""
Configuration for the Forex AI Trading Bot.
Edit these settings to match your broker account and trading preferences.
Supports both OANDA (Mac/Linux/Windows) and MT5 (Windows only).
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "forex_ai_bot" / "data"
MODEL_DIR = BASE_DIR / "forex_ai_bot" / "models"
LOG_DIR = BASE_DIR / "forex_ai_bot" / "logs"
CHECKPOINT_DIR = BASE_DIR / "forex_ai_bot" / "checkpoints"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# MT5 CONNECTION
# ============================================================
@dataclass
class MT5Config:
    """MetaTrader 5 connection settings."""
    login: int = 0              # Your MT5 account number
    password: str = ""          # Your MT5 password
    server: str = ""            # Your broker's MT5 server name
    path: str = ""              # Path to MT5 terminal (e.g. C:/Program Files/MetaTrader 5/terminal64.exe)
    timeout: int = 60000        # Connection timeout in ms


# ============================================================
# OANDA CONNECTION (Recommended for Mac)
# ============================================================
@dataclass
class OandaConfig:
    """OANDA REST API connection settings."""
    api_token: str = ""         # Your OANDA API token (generate at https://www.oanda.com/account/tpa/personal_token)
    account_id: str = ""        # Your OANDA account ID (e.g. "101-001-12345678-001")
    practice: bool = True       # True = demo account, False = live account


# ============================================================
# BROKER SELECTION
# ============================================================
# "oanda" = OANDA REST API (works on Mac/Linux/Windows)
# "mt5"   = MetaTrader 5 (Windows only)
BROKER = "oanda"


# ============================================================
# TRADING PAIRS
# ============================================================
@dataclass
class TradingPairsConfig:
    """Currency pairs to trade."""
    # Major pairs
    majors: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD"
    ])
    # Minor / Cross pairs
    minors: List[str] = field(default_factory=lambda: [
        "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",
        "EURAUD", "GBPAUD", "EURNZD", "AUDCAD",
        "CADJPY", "CHFJPY", "NZDJPY", "GBPCAD"
    ])
    # Exotic pairs
    exotics: List[str] = field(default_factory=lambda: [
        "USDTRY", "USDZAR", "USDMXN", "USDSEK",
        "USDNOK", "USDDKK", "USDSGD", "USDHKD",
        "USDPLN", "USDCZK", "USDHUF", "USDTHB"
    ])

    @property
    def all_pairs(self) -> List[str]:
        return self.majors + self.minors + self.exotics


# ============================================================
# TIMEFRAMES
# ============================================================
TIMEFRAMES = {
    "M1": 1,       # 1 minute
    "M5": 5,       # 5 minutes
    "M15": 15,     # 15 minutes
    "M30": 30,     # 30 minutes
    "H1": 60,      # 1 hour
    "H4": 240,     # 4 hours
    "D1": 1440,    # 1 day
}

# Primary timeframe for trading signals
PRIMARY_TIMEFRAME = "H1"
# Higher timeframes for trend confirmation
CONFIRMATION_TIMEFRAMES = ["H4", "D1"]


# ============================================================
# FEATURE ENGINEERING
# ============================================================
@dataclass
class FeatureConfig:
    """Settings for technical indicators and features."""
    # Moving averages
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 12, 21, 26, 50])

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # ADX
    adx_period: int = 14

    # Lookback for price features
    lookback_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])

    # Number of candles for historical data
    history_bars: int = 5000


# ============================================================
# ML MODEL SETTINGS
# ============================================================
@dataclass
class ModelConfig:
    """Settings for ML models."""
    # Train/test split
    test_size: float = 0.2
    validation_size: float = 0.1

    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_split: int = 10

    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8

    # Gradient Boosting
    gb_n_estimators: int = 200
    gb_max_depth: int = 6
    gb_learning_rate: float = 0.05

    # LSTM
    lstm_units: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_sequence_length: int = 60  # lookback window

    # Ensemble weights (will be optimized during training)
    # [RandomForest, XGBoost, GradientBoosting, LSTM]
    ensemble_weights: List[float] = field(default_factory=lambda: [0.25, 0.30, 0.20, 0.25])

    # Minimum confidence threshold for trading signals
    min_confidence: float = 0.60

    # Retrain interval (in hours)
    retrain_interval_hours: int = 24

    # Minimum samples before retraining
    min_retrain_samples: int = 100


# ============================================================
# RISK MANAGEMENT
# ============================================================
@dataclass
class RiskConfig:
    """Risk management settings."""
    # Maximum risk per trade (% of balance)
    max_risk_per_trade: float = 1.0      # 1% of balance

    # Maximum total exposure (% of balance)
    max_total_exposure: float = 5.0      # 5% of balance

    # Maximum number of simultaneous open trades
    max_open_trades: int = 5

    # Maximum trades per day
    max_trades_per_day: int = 10

    # Stop loss in ATR multiples
    stop_loss_atr_multiplier: float = 2.0

    # Take profit in ATR multiples
    take_profit_atr_multiplier: float = 3.0

    # Trailing stop activation (in pips profit)
    trailing_stop_activation: float = 30.0

    # Trailing stop distance (in pips)
    trailing_stop_distance: float = 15.0

    # Maximum drawdown before stopping (% of balance)
    max_drawdown_percent: float = 10.0

    # Maximum consecutive losses before pausing
    max_consecutive_losses: int = 5

    # Cooldown after max consecutive losses (in minutes)
    loss_cooldown_minutes: int = 60

    # Minimum lot size
    min_lot_size: float = 0.01

    # Maximum lot size
    max_lot_size: float = 1.0


# ============================================================
# TRADING SCHEDULE
# ============================================================
@dataclass
class ScheduleConfig:
    """Trading schedule (times in UTC)."""
    # Trading sessions
    trade_on_monday: bool = True
    trade_on_tuesday: bool = True
    trade_on_wednesday: bool = True
    trade_on_thursday: bool = True
    trade_on_friday: bool = True
    trade_on_saturday: bool = False
    trade_on_sunday: bool = False

    # Avoid trading during high-impact news (minutes before/after)
    news_avoidance_minutes: int = 30

    # Session hours (UTC) - bot is more aggressive during overlap periods
    london_open: int = 8
    london_close: int = 16
    ny_open: int = 13
    ny_close: int = 21
    tokyo_open: int = 0
    tokyo_close: int = 8


# ============================================================
# LOGGING
# ============================================================
@dataclass
class LogConfig:
    """Logging settings."""
    level: str = "INFO"
    log_trades: bool = True
    log_signals: bool = True
    log_model_performance: bool = True
    log_to_file: bool = True
    log_file: str = str(LOG_DIR / "trading_bot.log")


# ============================================================
# MASTER CONFIG
# ============================================================
@dataclass
class BotConfig:
    """Master configuration combining all settings."""
    mt5: MT5Config = field(default_factory=MT5Config)
    oanda: OandaConfig = field(default_factory=OandaConfig)
    pairs: TradingPairsConfig = field(default_factory=TradingPairsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # Broker: "oanda" or "mt5"
    broker: str = BROKER

    # Bot mode
    mode: str = "demo"  # "demo" or "live"

    # Paper trading (simulated, no real orders)
    paper_trading: bool = True


# Default config instance
config = BotConfig()
