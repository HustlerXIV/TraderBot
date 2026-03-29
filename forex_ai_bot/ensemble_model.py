"""
Ensemble ML Model
Combines Random Forest, XGBoost, Gradient Boosting, and LSTM
to generate robust trading signals with confidence scores.
"""

import logging
import pickle
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .config import ModelConfig, MODEL_DIR, CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model combining 4 ML models for Forex signal prediction.

    Models:
        1. Random Forest - Good at capturing non-linear patterns
        2. XGBoost - Excellent gradient boosting with regularization
        3. Gradient Boosting - Sklearn's implementation for diversity
        4. LSTM - Deep learning for sequential pattern recognition

    The ensemble uses weighted voting, where weights are optimized
    based on each model's recent performance.
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.scaler = StandardScaler()
        self.lstm_scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.performance_history = []

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize all sub-models."""
        # 1. Random Forest
        self.models["random_forest"] = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )

        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            self.models["xgboost"] = XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        else:
            logger.warning("XGBoost not available, using extra Random Forest")
            self.models["xgboost"] = RandomForestClassifier(
                n_estimators=300, max_depth=10, n_jobs=-1, random_state=43
            )

        # 3. Gradient Boosting
        self.models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=self.config.gb_n_estimators,
            max_depth=self.config.gb_max_depth,
            learning_rate=self.config.gb_learning_rate,
            subsample=0.8,
            random_state=42
        )

        # 4. LSTM (initialized during training)
        self.models["lstm"] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        X_lstm: Optional[np.ndarray] = None,
        y_lstm: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train all models in the ensemble.

        Args:
            X: Feature matrix (2D) for tree-based models
            y: Target labels (-1, 0, 1)
            feature_names: List of feature names
            X_lstm: Sequential data (3D) for LSTM
            y_lstm: Targets aligned with LSTM sequences

        Returns:
            Dict with training metrics for each model
        """
        self.feature_names = feature_names
        metrics = {}

        # Map labels: -1,0,1 -> 0,1,2 for classifiers
        y_mapped = y + 1  # Now: 0=SELL, 1=HOLD, 2=BUY

        # Train/test split (keeping temporal order)
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_mapped[:split_idx], y_mapped[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # --- Train tree-based models ---
        for name in ["random_forest", "xgboost", "gradient_boosting"]:
            logger.info(f"Training {name}...")
            try:
                self.models[name].fit(X_train_scaled, y_train)
                y_pred = self.models[name].predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                metrics[name] = {"accuracy": acc, "f1_score": f1}
                logger.info(f"  {name} -> Accuracy: {acc:.4f} | F1: {f1:.4f}")
            except Exception as e:
                logger.error(f"  Failed to train {name}: {e}")
                metrics[name] = {"accuracy": 0, "f1_score": 0}

        # --- Train LSTM ---
        if TENSORFLOW_AVAILABLE and X_lstm is not None:
            logger.info("Training LSTM...")
            try:
                metrics["lstm"] = self._train_lstm(X_lstm, y_lstm + 1)
            except Exception as e:
                logger.error(f"  Failed to train LSTM: {e}")
                metrics["lstm"] = {"accuracy": 0, "f1_score": 0}
        else:
            logger.info("LSTM skipped (TensorFlow not available or no sequential data)")
            metrics["lstm"] = {"accuracy": 0, "f1_score": 0}

        # Optimize ensemble weights based on performance
        self._optimize_weights(metrics)

        self.is_trained = True
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "weights": self.config.ensemble_weights.copy()
        })

        logger.info(f"Ensemble weights: {dict(zip(self.models.keys(), self.config.ensemble_weights))}")
        return metrics

    def predict(self, X: np.ndarray, X_lstm: Optional[np.ndarray] = None) -> Tuple[int, float, Dict]:
        """
        Generate ensemble prediction.

        Args:
            X: Feature vector(s) - 2D array
            X_lstm: Sequential data for LSTM - 3D array

        Returns:
            (signal, confidence, details)
            signal: -1=SELL, 0=HOLD, 1=BUY
            confidence: 0-1 probability
            details: Per-model predictions
        """
        if not self.is_trained:
            logger.warning("Models not trained yet")
            return 0, 0.0, {}

        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        predictions = {}
        probabilities = {}

        # Get predictions from each model
        weights = self.config.ensemble_weights
        model_names = ["random_forest", "xgboost", "gradient_boosting", "lstm"]

        for i, name in enumerate(model_names[:3]):
            try:
                pred = self.models[name].predict(X_scaled[-1:])
                prob = self.models[name].predict_proba(X_scaled[-1:])
                predictions[name] = int(pred[0]) - 1  # Map back: 0,1,2 -> -1,0,1
                probabilities[name] = prob[0].tolist()
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                predictions[name] = 0
                probabilities[name] = [0.33, 0.34, 0.33]

        # LSTM prediction
        if self.models["lstm"] is not None and X_lstm is not None and TENSORFLOW_AVAILABLE:
            try:
                lstm_pred = self.models["lstm"].predict(X_lstm[-1:], verbose=0)
                lstm_class = np.argmax(lstm_pred[0])
                predictions["lstm"] = int(lstm_class) - 1
                probabilities["lstm"] = lstm_pred[0].tolist()
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
                predictions["lstm"] = 0
                probabilities["lstm"] = [0.33, 0.34, 0.33]
        else:
            predictions["lstm"] = 0
            probabilities["lstm"] = [0.33, 0.34, 0.33]

        # Weighted ensemble voting
        weighted_probs = np.zeros(3)  # [SELL, HOLD, BUY]
        total_weight = 0

        for i, name in enumerate(model_names):
            if name in probabilities:
                prob = np.array(probabilities[name])
                if len(prob) == 3:
                    weighted_probs += weights[i] * prob
                    total_weight += weights[i]

        if total_weight > 0:
            weighted_probs /= total_weight

        # Final signal
        final_class = np.argmax(weighted_probs)
        confidence = float(weighted_probs[final_class])
        signal = int(final_class) - 1  # Map: 0,1,2 -> -1,0,1

        details = {
            "predictions": predictions,
            "probabilities": probabilities,
            "weighted_probs": {
                "sell": float(weighted_probs[0]),
                "hold": float(weighted_probs[1]),
                "buy": float(weighted_probs[2])
            },
            "signal": signal,
            "confidence": confidence,
        }

        return signal, confidence, details

    def _train_lstm(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the LSTM model."""
        seq_len = self.config.lstm_sequence_length

        # Split
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # One-hot encode targets
        n_classes = 3
        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        y_test_cat = to_categorical(y_test, num_classes=n_classes)

        # Build model
        n_features = X_train.shape[2]
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True,
                 input_shape=(X_train.shape[1], n_features)),
            Dropout(self.config.lstm_dropout),
            BatchNormalization(),
            LSTM(self.config.lstm_units // 2, return_sequences=False),
            Dropout(self.config.lstm_dropout),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(n_classes, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]

        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.15,
            epochs=self.config.lstm_epochs,
            batch_size=self.config.lstm_batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        f1 = f1_score(y_test, y_pred, average="weighted")

        self.models["lstm"] = model
        logger.info(f"  LSTM -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

        return {"accuracy": acc, "f1_score": f1}

    def _optimize_weights(self, metrics: Dict):
        """Optimize ensemble weights based on model performance."""
        model_names = ["random_forest", "xgboost", "gradient_boosting", "lstm"]
        scores = []

        for name in model_names:
            if name in metrics:
                # Combine accuracy and F1 for weight calculation
                score = metrics[name].get("f1_score", 0) * 0.7 + metrics[name].get("accuracy", 0) * 0.3
                scores.append(max(score, 0.01))  # Minimum weight
            else:
                scores.append(0.01)

        # Normalize to sum to 1
        total = sum(scores)
        self.config.ensemble_weights = [s / total for s in scores]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance across tree-based models."""
        if not self.is_trained or not self.feature_names:
            return {}

        importance = np.zeros(len(self.feature_names))
        count = 0

        for name in ["random_forest", "xgboost", "gradient_boosting"]:
            model = self.models.get(name)
            if model and hasattr(model, "feature_importances_"):
                importance += model.feature_importances_
                count += 1

        if count > 0:
            importance /= count

        return dict(sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))

    def save(self, path: Path = None):
        """Save all models and scalers to disk."""
        save_dir = path or MODEL_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save tree-based models
        for name in ["random_forest", "xgboost", "gradient_boosting"]:
            if self.models.get(name) is not None:
                filepath = save_dir / f"{name}_{timestamp}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(self.models[name], f)

        # Save LSTM
        if self.models.get("lstm") is not None and TENSORFLOW_AVAILABLE:
            lstm_path = save_dir / f"lstm_{timestamp}.keras"
            self.models["lstm"].save(str(lstm_path))

        # Save scalers
        with open(save_dir / f"scaler_{timestamp}.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "feature_names": self.feature_names,
            "ensemble_weights": self.config.ensemble_weights,
            "performance_history": self.performance_history[-10:],  # Last 10
        }
        with open(save_dir / f"metadata_{timestamp}.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save a "latest" pointer
        with open(save_dir / "latest.txt", "w") as f:
            f.write(timestamp)

        logger.info(f"Models saved to {save_dir} (timestamp: {timestamp})")

    def load(self, path: Path = None) -> bool:
        """Load the latest saved models."""
        load_dir = path or MODEL_DIR
        load_dir = Path(load_dir)

        # Find latest timestamp
        latest_file = load_dir / "latest.txt"
        if not latest_file.exists():
            logger.warning("No saved models found")
            return False

        timestamp = latest_file.read_text().strip()
        logger.info(f"Loading models from timestamp: {timestamp}")

        try:
            # Load tree-based models
            for name in ["random_forest", "xgboost", "gradient_boosting"]:
                filepath = load_dir / f"{name}_{timestamp}.pkl"
                if filepath.exists():
                    with open(filepath, "rb") as f:
                        self.models[name] = pickle.load(f)

            # Load LSTM
            if TENSORFLOW_AVAILABLE:
                lstm_path = load_dir / f"lstm_{timestamp}.keras"
                if lstm_path.exists():
                    self.models["lstm"] = load_model(str(lstm_path))

            # Load scaler
            scaler_path = load_dir / f"scaler_{timestamp}.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            # Load metadata
            meta_path = load_dir / f"metadata_{timestamp}.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                self.feature_names = metadata.get("feature_names", [])
                self.config.ensemble_weights = metadata.get("ensemble_weights", [0.25] * 4)
                self.performance_history = metadata.get("performance_history", [])

            self.is_trained = True
            logger.info("Models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
