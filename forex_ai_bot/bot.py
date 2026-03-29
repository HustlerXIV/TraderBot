"""
Main Bot Orchestrator
The central controller that ties everything together:
Broker connection, feature engineering, ML models,
strategy, risk management, execution, and learning.

Supports OANDA (Mac/Linux/Windows) and MT5 (Windows).
"""

import logging
import time
import signal as sig
import sys
from datetime import datetime
from typing import Optional

from .config import BotConfig, config as default_config
from .mt5_connector import MT5Connector
from .oanda_connector import OandaConnector
from .feature_engine import FeatureEngine
from .ensemble_model import EnsembleModel
from .risk_manager import RiskManager
from .strategy import TradingStrategy
from .execution_engine import ExecutionEngine
from .continuous_learner import ContinuousLearner

logger = logging.getLogger(__name__)


class ForexAIBot:
    """
    Main Forex AI Trading Bot.

    Lifecycle:
    1. Initialize all components
    2. Connect to MT5
    3. Train initial models (or load existing)
    4. Enter main trading loop:
       a. Scan for signals
       b. Execute qualifying trades
       c. Manage open positions
       d. Check if retraining needed
       e. Sleep and repeat
    """

    def __init__(self, config: BotConfig = None):
        self.config = config or default_config
        self.running = False
        self.cycle_count = 0

        # Setup logging
        self._setup_logging()

        # Initialize components
        logger.info("=" * 60)
        logger.info("FOREX AI TRADING BOT v1.0")
        logger.info("=" * 60)

        # Choose broker connector
        if self.config.broker == "oanda":
            self.connector = OandaConnector(self.config)
            logger.info("Broker: OANDA REST API")
        else:
            self.connector = MT5Connector(self.config)
            logger.info("Broker: MetaTrader 5")
        self.feature_engine = FeatureEngine(self.config.features)
        self.model = EnsembleModel(self.config.model)
        self.risk_manager = RiskManager(self.config.risk)
        self.strategy = TradingStrategy(
            self.config, self.connector, self.feature_engine,
            self.model, self.risk_manager
        )
        self.executor = ExecutionEngine(
            self.config, self.connector, self.risk_manager
        )
        self.learner = ContinuousLearner(
            self.config, self.connector, self.feature_engine, self.model
        )

        logger.info(f"Mode: {'PAPER TRADING' if self.config.paper_trading else self.config.mode.upper()}")
        logger.info(f"Pairs: {len(self.config.pairs.all_pairs)} total")
        logger.info(f"Min confidence: {self.config.model.min_confidence:.0%}")

    def start(self):
        """Start the trading bot."""
        logger.info("Starting bot...")

        # Register signal handlers for graceful shutdown
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)

        # Step 1: Connect to MT5
        connected = self.connector.connect()
        if connected:
            account = self.connector.get_account_info()
            logger.info(f"Account balance: {account.get('balance', 'N/A')} {account.get('currency', '')}")
        else:
            logger.info("Running in simulation mode (MT5 not connected)")

        # Step 2: Load or train models
        if not self.model.load():
            logger.info("No saved models found. Starting initial training...")
            self._initial_training()
        else:
            logger.info("Loaded pre-trained models")

        # Step 3: Enter main loop
        self.running = True
        self._main_loop()

    def stop(self):
        """Stop the trading bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False

        # Save models
        if self.model.is_trained:
            self.model.save()
            logger.info("Models saved")

        # Close all positions if configured
        # (commented out for safety - uncomment if you want auto-close)
        # self.executor.close_all_positions("Bot shutdown")

        # Disconnect
        self.connector.disconnect()

        # Final stats
        stats = self.risk_manager.get_stats()
        logger.info(f"Final stats: {stats}")
        logger.info("Bot stopped")

    def run_single_cycle(self):
        """Run a single trading cycle (useful for testing)."""
        self.connector.connect()
        if not self.model.is_trained:
            self._initial_training()
        self._trading_cycle()
        self.connector.disconnect()

    def backtest(self, symbol: str = "EURUSD", bars: int = 2000) -> dict:
        """
        Run a simple backtest on historical data.

        Returns:
            Backtest results dict
        """
        logger.info(f"Backtesting on {symbol} ({bars} bars)...")

        # Get data
        self.connector.connect()
        df = self.connector.get_historical_data(symbol, "H1", bars)
        self.connector.disconnect()

        if df is None:
            return {"error": "No data"}

        # Feature engineering + labels
        featured = self.feature_engine.compute_all_features(df)
        labeled = self.feature_engine.create_labels(featured)

        if labeled.empty:
            return {"error": "No labeled data"}

        # Train on first 70%, test on last 30%
        split = int(len(labeled) * 0.7)
        train_data = labeled.iloc[:split]
        test_data = labeled.iloc[split:]

        # Train
        X_train, y_train, feature_names = self.feature_engine.prepare_ml_data(train_data)
        metrics = self.model.train(X_train, y_train, feature_names)

        # Simulate trading on test period
        X_test, y_test, _ = self.feature_engine.prepare_ml_data(test_data)
        trades = []
        balance = 10000.0
        wins = 0
        losses = 0

        for i in range(len(X_test)):
            signal, confidence, details = self.model.predict(X_test[i])

            if signal == 0 or confidence < self.config.model.min_confidence:
                continue

            # Simple PnL simulation
            actual = y_test[i]
            if signal == actual:
                pnl = balance * 0.01  # 1% gain
                wins += 1
            elif actual == 0:
                pnl = -balance * 0.002  # Small loss on HOLD
                losses += 1
            else:
                pnl = -balance * 0.01  # 1% loss
                losses += 1

            balance += pnl
            trades.append({
                "index": i,
                "signal": signal,
                "actual": int(actual),
                "confidence": float(confidence),
                "pnl": pnl,
                "balance": balance
            })

        total_trades = wins + losses
        results = {
            "symbol": symbol,
            "period": f"{test_data.index[0]} to {test_data.index[-1]}",
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades * 100 if total_trades > 0 else 0,
            "final_balance": round(balance, 2),
            "return_pct": round((balance - 10000) / 10000 * 100, 2),
            "model_metrics": metrics,
            "trades": trades[-20:],  # Last 20 trades for review
        }

        logger.info(
            f"Backtest complete: {total_trades} trades | "
            f"Win rate: {results['win_rate']:.1f}% | "
            f"Return: {results['return_pct']}%"
        )
        return results

    # ---- Private methods ----

    def _main_loop(self):
        """Main trading loop."""
        logger.info("Entering main trading loop...")

        while self.running:
            try:
                self._trading_cycle()

                # Check if retraining needed
                should_retrain, reason = self.learner.should_retrain()
                if should_retrain:
                    logger.info(f"Retraining triggered: {reason}")
                    self.learner.train_models()

                # Sleep between cycles (5 minutes for H1 timeframe)
                sleep_seconds = 300
                logger.info(f"Sleeping {sleep_seconds}s until next cycle...")
                for _ in range(sleep_seconds):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 min on error

    def _trading_cycle(self):
        """Run a single trading cycle."""
        self.cycle_count += 1
        logger.info(f"\n--- Trading Cycle #{self.cycle_count} | {datetime.now()} ---")

        # Check trading schedule
        if not self._is_trading_time():
            logger.info("Outside trading hours, skipping")
            return

        # Manage existing positions
        self.executor.manage_positions()
        self.executor.check_closed_trades()

        # Scan for new signals
        signals = self.strategy.scan_all_pairs()

        # Execute top signals
        for signal in signals:
            result = self.executor.execute_signal(signal)
            if result:
                logger.info(f"Executed: {signal.symbol} {signal.signal_name}")

        # Portfolio summary
        summary = self.executor.get_portfolio_summary()
        logger.info(
            f"Portfolio: {summary['open_positions']} positions | "
            f"Unrealized PnL: {summary['unrealized_pnl']:.2f}"
        )

    def _initial_training(self):
        """Perform initial model training."""
        logger.info("Starting initial model training...")

        # Use a subset of pairs for faster initial training
        training_pairs = self.config.pairs.majors[:5]
        datasets = self.learner.collect_training_data(symbols=training_pairs)

        if datasets:
            metrics = self.learner.train_models(datasets, validate=False)
            logger.info(f"Initial training complete: {metrics}")
        else:
            logger.warning("Could not collect training data")

    def _is_trading_time(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.utcnow()
        weekday = now.weekday()  # 0=Monday

        schedule = self.config.schedule
        day_flags = [
            schedule.trade_on_monday,
            schedule.trade_on_tuesday,
            schedule.trade_on_wednesday,
            schedule.trade_on_thursday,
            schedule.trade_on_friday,
            schedule.trade_on_saturday,
            schedule.trade_on_sunday,
        ]

        return day_flags[weekday]

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.log

        handlers = [logging.StreamHandler(sys.stdout)]
        if log_config.log_to_file:
            handlers.append(logging.FileHandler(log_config.log_file))

        logging.basicConfig(
            level=getattr(logging, log_config.level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True
        )
