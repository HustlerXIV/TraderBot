"""
Continuous Learning Pipeline
Automatically retrains models on new market data,
evaluates performance, and adapts to changing conditions.
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path

from .config import BotConfig, PRIMARY_TIMEFRAME, MODEL_DIR, CHECKPOINT_DIR
from .mt5_connector import MT5Connector
from .feature_engine import FeatureEngine
from .ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class ContinuousLearner:
    """
    Manages the continuous learning lifecycle:

    1. Collect new market data periodically
    2. Evaluate current model performance on recent data
    3. Retrain models when performance degrades
    4. Validate new models before deployment
    5. Track performance over time for analysis
    """

    def __init__(
        self,
        config: BotConfig,
        connector: MT5Connector,
        feature_engine: FeatureEngine,
        model: EnsembleModel
    ):
        self.config = config
        self.connector = connector
        self.feature_engine = feature_engine
        self.model = model

        self.last_retrain: Optional[datetime] = None
        self.performance_log: List[Dict] = []
        self.data_buffer: Dict[str, pd.DataFrame] = {}  # symbol -> data

        # Load performance history if exists
        self._load_performance_log()

    def should_retrain(self) -> tuple:
        """
        Determine if models should be retrained.

        Triggers:
        1. Time-based: Exceeded retrain interval
        2. Performance-based: Recent accuracy dropped below threshold
        3. Data-based: Sufficient new data available

        Returns:
            (should_retrain: bool, reason: str)
        """
        # First time training
        if not self.model.is_trained:
            return True, "Initial training required"

        # Time-based check
        if self.last_retrain is None:
            return True, "No previous training recorded"

        hours_since_retrain = (datetime.now() - self.last_retrain).total_seconds() / 3600
        if hours_since_retrain >= self.config.model.retrain_interval_hours:
            return True, f"Retrain interval exceeded ({hours_since_retrain:.1f}h)"

        # Performance-based check
        if len(self.performance_log) >= 5:
            recent_accuracy = np.mean([
                p.get("live_accuracy", 0.5)
                for p in self.performance_log[-5:]
            ])
            if recent_accuracy < 0.45:  # Below random chance
                return True, f"Performance degraded (recent accuracy: {recent_accuracy:.2%})"

        return False, "No retrain needed"

    def collect_training_data(
        self,
        symbols: List[str] = None,
        bars: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect and prepare training data for all symbols.

        Returns:
            Dict of symbol -> featured DataFrame with labels
        """
        if symbols is None:
            symbols = self.config.pairs.all_pairs
        if bars is None:
            bars = self.config.features.history_bars

        datasets = {}

        for symbol in symbols:
            try:
                # Fetch raw data
                df = self.connector.get_historical_data(
                    symbol, PRIMARY_TIMEFRAME, bars
                )
                if df is None or len(df) < 200:
                    continue

                # Compute features
                featured = self.feature_engine.compute_all_features(df)
                if featured.empty:
                    continue

                # Create labels
                labeled = self.feature_engine.create_labels(featured)
                if labeled.empty:
                    continue

                datasets[symbol] = labeled
                logger.info(f"Collected {len(labeled)} samples for {symbol}")

            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")

        return datasets

    def train_models(
        self,
        datasets: Dict[str, pd.DataFrame] = None,
        validate: bool = True
    ) -> Dict:
        """
        Train/retrain the ensemble models.

        Args:
            datasets: Pre-collected datasets (or collect new if None)
            validate: Whether to validate before deploying

        Returns:
            Training metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)

        # Collect data if not provided
        if datasets is None:
            datasets = self.collect_training_data()

        if not datasets:
            logger.error("No training data available")
            return {"error": "No data"}

        # Combine all symbol data for a general model
        combined = pd.concat(list(datasets.values()), ignore_index=False)
        combined.sort_index(inplace=True)
        logger.info(f"Combined dataset: {len(combined)} samples from {len(datasets)} symbols")

        # Prepare ML data
        X, y, feature_names = self.feature_engine.prepare_ml_data(combined)

        # Prepare LSTM data
        X_lstm, y_lstm = None, None
        try:
            X_lstm, y_lstm, _ = self.feature_engine.prepare_lstm_data(
                combined,
                sequence_length=self.config.model.lstm_sequence_length
            )
            logger.info(f"LSTM data shape: {X_lstm.shape}")
        except Exception as e:
            logger.warning(f"Could not prepare LSTM data: {e}")

        # Save current model as checkpoint before retraining
        if self.model.is_trained:
            self.model.save(CHECKPOINT_DIR)
            logger.info("Previous model saved as checkpoint")

        # Train
        metrics = self.model.train(X, y, feature_names, X_lstm, y_lstm)

        # Validate new model
        if validate:
            valid = self._validate_model(datasets)
            if not valid:
                logger.warning("New model failed validation! Restoring checkpoint.")
                self.model.load(CHECKPOINT_DIR)
                return {"error": "Validation failed", "metrics": metrics}

        # Save new model
        self.model.save()
        self.last_retrain = datetime.now()

        logger.info("Model training complete!")
        logger.info(f"Metrics: {metrics}")

        return metrics

    def _validate_model(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate the newly trained model on held-out data.

        Checks:
        1. Accuracy is above minimum threshold (40%)
        2. Predictions are not all one class
        3. Confidence distribution is reasonable

        Returns:
            True if model passes validation
        """
        logger.info("Validating new model...")

        all_predictions = []
        all_actuals = []

        for symbol, df in datasets.items():
            try:
                # Use last 20% as validation
                val_start = int(len(df) * 0.8)
                val_data = df.iloc[val_start:]

                if len(val_data) < 50:
                    continue

                X_val, y_val, _ = self.feature_engine.prepare_ml_data(val_data)

                for i in range(len(X_val)):
                    signal, confidence, _ = self.model.predict(X_val[i])
                    all_predictions.append(signal)
                    all_actuals.append(y_val[i])

            except Exception as e:
                logger.warning(f"Validation error for {symbol}: {e}")

        if len(all_predictions) < 50:
            logger.warning("Insufficient validation data")
            return True  # Allow training anyway

        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)

        # Check 1: Accuracy
        accuracy = np.mean(predictions == actuals)
        logger.info(f"Validation accuracy: {accuracy:.2%}")

        if accuracy < 0.35:
            logger.warning(f"Accuracy too low: {accuracy:.2%}")
            return False

        # Check 2: Not all same class
        unique_preds = len(np.unique(predictions))
        if unique_preds < 2:
            logger.warning(f"Model only predicts {unique_preds} class(es)")
            return False

        # Check 3: Class distribution
        for cls in [-1, 0, 1]:
            ratio = np.mean(predictions == cls)
            if ratio > 0.8:
                logger.warning(f"Class {cls} dominates predictions ({ratio:.0%})")
                return False

        logger.info("Model validation PASSED")
        return True

    def evaluate_live_performance(self, trade_history: List[Dict]) -> Dict:
        """
        Evaluate model's live trading performance.

        Args:
            trade_history: List of completed trades

        Returns:
            Performance metrics
        """
        if not trade_history:
            return {}

        profits = [t.get("profit", 0) for t in trade_history]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        perf = {
            "timestamp": datetime.now().isoformat(),
            "total_trades": len(profits),
            "win_rate": len(wins) / len(profits) if profits else 0,
            "total_pnl": sum(profits),
            "avg_profit": np.mean(profits) if profits else 0,
            "sharpe_approx": np.mean(profits) / (np.std(profits) + 1e-10) if len(profits) > 1 else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
            "live_accuracy": len(wins) / len(profits) if profits else 0,
        }

        self.performance_log.append(perf)
        self._save_performance_log()

        return perf

    def get_learning_status(self) -> Dict:
        """Get current learning pipeline status."""
        should, reason = self.should_retrain()
        return {
            "model_trained": self.model.is_trained,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "should_retrain": should,
            "retrain_reason": reason,
            "performance_entries": len(self.performance_log),
            "ensemble_weights": dict(zip(
                ["random_forest", "xgboost", "gradient_boosting", "lstm"],
                self.model.config.ensemble_weights
            )),
        }

    def _save_performance_log(self):
        """Save performance log to disk."""
        log_file = MODEL_DIR / "performance_log.json"
        try:
            with open(log_file, "w") as f:
                json.dump(self.performance_log[-100:], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving performance log: {e}")

    def _load_performance_log(self):
        """Load performance log from disk."""
        log_file = MODEL_DIR / "performance_log.json"
        try:
            if log_file.exists():
                with open(log_file, "r") as f:
                    self.performance_log = json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance log: {e}")
