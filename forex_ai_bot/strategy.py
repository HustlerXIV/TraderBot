"""
Trading Strategy & Signal Generator
Combines ML ensemble predictions with technical analysis
confirmation to produce actionable trading signals.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

from .config import BotConfig, PRIMARY_TIMEFRAME, CONFIRMATION_TIMEFRAMES
from .mt5_connector import MT5Connector
from .feature_engine import FeatureEngine
from .ensemble_model import EnsembleModel
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class TradingSignal:
    """Represents a trading signal with all necessary information."""

    def __init__(
        self,
        symbol: str,
        signal: int,          # -1=SELL, 0=HOLD, 1=BUY
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,
        timeframe: str,
        details: Dict = None
    ):
        self.symbol = symbol
        self.signal = signal
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.lot_size = lot_size
        self.timeframe = timeframe
        self.details = details or {}
        self.timestamp = datetime.now()

    @property
    def signal_name(self) -> str:
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(self.signal, "HOLD")

    def __repr__(self):
        return (
            f"Signal({self.symbol} {self.signal_name} @ {self.entry_price} | "
            f"SL: {self.stop_loss} | TP: {self.take_profit} | "
            f"Lots: {self.lot_size} | Conf: {self.confidence:.2%})"
        )


class TradingStrategy:
    """
    Orchestrates signal generation by combining:
    1. Multi-timeframe analysis
    2. Ensemble ML predictions
    3. Technical confirmation filters
    4. Risk management sizing
    """

    def __init__(
        self,
        config: BotConfig,
        connector: MT5Connector,
        feature_engine: FeatureEngine,
        model: EnsembleModel,
        risk_manager: RiskManager
    ):
        self.config = config
        self.connector = connector
        self.feature_engine = feature_engine
        self.model = model
        self.risk_manager = risk_manager
        self.signal_history: List[TradingSignal] = []

    def analyze_pair(self, symbol: str) -> Optional[TradingSignal]:
        """
        Perform full analysis on a currency pair.

        Steps:
        1. Fetch data for primary + confirmation timeframes
        2. Compute features
        3. Get ML ensemble prediction
        4. Apply confirmation filters
        5. Calculate risk parameters
        6. Generate final signal

        Returns:
            TradingSignal or None if no valid signal
        """
        logger.info(f"Analyzing {symbol}...")

        # 1. Fetch primary timeframe data
        primary_df = self.connector.get_historical_data(
            symbol, PRIMARY_TIMEFRAME, self.config.features.history_bars
        )
        if primary_df is None or len(primary_df) < 200:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # 2. Compute features
        featured_df = self.feature_engine.compute_all_features(primary_df)
        if featured_df.empty:
            return None

        # 3. Prepare data and get ML prediction
        X, _, feature_names = self.feature_engine.prepare_ml_data(
            featured_df.assign(target=0)  # Dummy target for feature extraction
        )

        # Prepare LSTM data if available
        X_lstm = None
        try:
            X_lstm_full, _, _ = self.feature_engine.prepare_lstm_data(
                featured_df.assign(target=0),
                sequence_length=self.config.model.lstm_sequence_length
            )
            if len(X_lstm_full) > 0:
                X_lstm = X_lstm_full
        except Exception:
            pass

        # Get ensemble prediction
        signal, confidence, details = self.model.predict(X[-1], X_lstm)

        # 4. Apply confirmation filters
        confirmed, filter_details = self._apply_filters(featured_df, signal, symbol)

        if not confirmed:
            logger.info(f"{symbol}: Signal {signal} rejected by filters - {filter_details}")
            return None

        # Check minimum confidence
        if confidence < self.config.model.min_confidence:
            logger.info(f"{symbol}: Confidence too low ({confidence:.2%})")
            return None

        # 5. Calculate risk parameters
        symbol_info = self.connector.get_symbol_info(symbol)
        if not symbol_info:
            return None

        price_info = self.connector.get_current_price(symbol)
        if not price_info:
            return None

        current_price = price_info["ask"] if signal == 1 else price_info["bid"]
        atr = featured_df["atr"].iloc[-1] if "atr" in featured_df.columns else 0.001

        stop_loss = self.risk_manager.calculate_stop_loss(
            signal, current_price, atr, symbol_info
        )
        take_profit = self.risk_manager.calculate_take_profit(
            signal, current_price, atr, symbol_info
        )

        # Calculate SL in pips for position sizing
        point = symbol_info.get("point", 0.00001)
        sl_pips = abs(current_price - stop_loss) / (point * 10)

        account_info = self.connector.get_account_info()
        lot_size = self.risk_manager.calculate_position_size(
            account_info, sl_pips, symbol_info
        )

        # 6. Create signal
        trading_signal = TradingSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            timeframe=PRIMARY_TIMEFRAME,
            details={
                **details,
                "filters": filter_details,
                "atr": float(atr),
                "sl_pips": float(sl_pips),
            }
        )

        self.signal_history.append(trading_signal)
        logger.info(f"Signal generated: {trading_signal}")
        return trading_signal

    def scan_all_pairs(self) -> List[TradingSignal]:
        """Scan all configured pairs and return valid signals."""
        signals = []
        pairs = self.config.pairs.all_pairs

        logger.info(f"Scanning {len(pairs)} currency pairs...")

        for symbol in pairs:
            try:
                signal = self.analyze_pair(symbol)
                if signal and signal.signal != 0:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(f"Found {len(signals)} valid signals out of {len(pairs)} pairs")
        return signals

    def _apply_filters(
        self,
        df: pd.DataFrame,
        signal: int,
        symbol: str
    ) -> Tuple[bool, Dict]:
        """
        Apply confirmation filters to validate the ML signal.

        Filters:
        1. Trend alignment (MA-based)
        2. RSI not in extreme zone against signal
        3. ADX showing trend strength
        4. Volume confirmation
        5. Higher timeframe alignment

        Returns:
            (confirmed: bool, details: Dict)
        """
        if signal == 0:
            return False, {"reason": "HOLD signal"}

        filters_passed = 0
        total_filters = 5
        details = {}

        latest = df.iloc[-1]

        # Filter 1: Trend alignment (50 SMA direction)
        if "sma_50" in df.columns and "sma_20" in df.columns:
            trend_up = latest["sma_20"] > latest["sma_50"]
            if (signal == 1 and trend_up) or (signal == -1 and not trend_up):
                filters_passed += 1
                details["trend"] = "aligned"
            else:
                details["trend"] = "misaligned"
        else:
            filters_passed += 1  # Skip if not available
            details["trend"] = "skipped"

        # Filter 2: RSI not extreme against signal
        if "rsi" in df.columns:
            rsi = latest["rsi"]
            if signal == 1 and rsi < 75:  # Don't buy if extremely overbought
                filters_passed += 1
                details["rsi"] = f"OK ({rsi:.1f})"
            elif signal == -1 and rsi > 25:  # Don't sell if extremely oversold
                filters_passed += 1
                details["rsi"] = f"OK ({rsi:.1f})"
            else:
                details["rsi"] = f"extreme ({rsi:.1f})"
        else:
            filters_passed += 1
            details["rsi"] = "skipped"

        # Filter 3: ADX showing trend
        if "adx" in df.columns:
            adx = latest["adx"]
            if adx > 20:  # Trending market
                filters_passed += 1
                details["adx"] = f"trending ({adx:.1f})"
            else:
                details["adx"] = f"ranging ({adx:.1f})"
        else:
            filters_passed += 1
            details["adx"] = "skipped"

        # Filter 4: Volume confirmation
        if "volume_ratio" in df.columns:
            vol_ratio = latest["volume_ratio"]
            if vol_ratio > 0.8:  # Above average volume
                filters_passed += 1
                details["volume"] = f"confirmed ({vol_ratio:.2f}x)"
            else:
                details["volume"] = f"low ({vol_ratio:.2f}x)"
        else:
            filters_passed += 1
            details["volume"] = "skipped"

        # Filter 5: MACD alignment
        if "macd_histogram" in df.columns:
            macd_hist = latest["macd_histogram"]
            if (signal == 1 and macd_hist > 0) or (signal == -1 and macd_hist < 0):
                filters_passed += 1
                details["macd"] = "aligned"
            else:
                details["macd"] = "misaligned"
        else:
            filters_passed += 1
            details["macd"] = "skipped"

        # Need at least 3 out of 5 filters to pass
        min_required = 3
        confirmed = filters_passed >= min_required
        details["passed"] = f"{filters_passed}/{total_filters}"
        details["required"] = min_required

        return confirmed, details

    def _check_higher_timeframe(self, symbol: str, signal: int) -> bool:
        """Check if higher timeframe confirms the signal direction."""
        for tf in CONFIRMATION_TIMEFRAMES:
            try:
                htf_data = self.connector.get_historical_data(symbol, tf, 200)
                if htf_data is None or len(htf_data) < 50:
                    continue

                htf_featured = self.feature_engine.compute_all_features(htf_data)
                if htf_featured.empty:
                    continue

                latest = htf_featured.iloc[-1]

                # Check trend direction on higher TF
                if "sma_20" in htf_featured.columns and "sma_50" in htf_featured.columns:
                    htf_trend_up = latest["sma_20"] > latest["sma_50"]
                    if (signal == 1 and not htf_trend_up) or (signal == -1 and htf_trend_up):
                        return False  # Higher TF disagrees

            except Exception:
                continue

        return True  # No contradiction found
