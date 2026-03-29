"""
Feature Engineering Pipeline
Computes technical indicators and transforms raw OHLCV data
into ML-ready features for the ensemble models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from .config import FeatureConfig

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Computes technical indicators and ML features from OHLCV data."""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and features.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with all features added
        """
        if df is None or df.empty:
            logger.warning("Empty dataframe passed to feature engine")
            return df

        data = df.copy()

        # Price-based features
        data = self._add_price_features(data)

        # Moving averages
        data = self._add_moving_averages(data)

        # RSI
        data = self._add_rsi(data)

        # MACD
        data = self._add_macd(data)

        # Bollinger Bands
        data = self._add_bollinger_bands(data)

        # ATR (Average True Range)
        data = self._add_atr(data)

        # Stochastic Oscillator
        data = self._add_stochastic(data)

        # ADX (Average Directional Index)
        data = self._add_adx(data)

        # Volume features
        data = self._add_volume_features(data)

        # Candlestick patterns
        data = self._add_candlestick_features(data)

        # Time-based features
        data = self._add_time_features(data)

        # Momentum features
        data = self._add_momentum_features(data)

        # Volatility features
        data = self._add_volatility_features(data)

        # Drop NaN rows from indicator calculations
        data.dropna(inplace=True)

        logger.info(f"Computed {len(data.columns)} features | {len(data)} rows")
        return data

    def create_labels(
        self,
        df: pd.DataFrame,
        lookahead: int = 5,
        threshold: float = 0.001
    ) -> pd.DataFrame:
        """
        Create target labels for supervised learning.

        Labels:
            1  = BUY  (price goes up by > threshold within lookahead periods)
            -1 = SELL (price goes down by > threshold within lookahead periods)
            0  = HOLD (no significant movement)

        Args:
            df: DataFrame with at least 'close' column
            lookahead: Number of periods to look ahead
            threshold: Minimum price change ratio for a signal

        Returns:
            DataFrame with 'target' column added
        """
        data = df.copy()

        # Future return
        future_return = data["close"].shift(-lookahead) / data["close"] - 1

        # Create labels
        data["target"] = 0  # HOLD
        data.loc[future_return > threshold, "target"] = 1    # BUY
        data.loc[future_return < -threshold, "target"] = -1  # SELL

        # Drop last rows where we can't compute future return
        data = data.iloc[:-lookahead]

        logger.info(
            f"Labels created | BUY: {(data['target']==1).sum()} | "
            f"SELL: {(data['target']==-1).sum()} | "
            f"HOLD: {(data['target']==0).sum()}"
        )
        return data

    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target arrays for ML training.

        Returns:
            X (features), y (targets), feature_names
        """
        # Columns to exclude from features
        exclude = ["open", "high", "low", "close", "volume", "spread", target_col]
        feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.int32)

        # Replace any remaining inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, feature_cols

    def prepare_lstm_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequential data for LSTM model.

        Returns:
            X (3D: samples x timesteps x features), y (targets), feature_names
        """
        exclude = ["open", "high", "low", "close", "volume", "spread", target_col]
        feature_cols = [c for c in df.columns if c not in exclude]

        data = df[feature_cols].values.astype(np.float32)
        targets = df[target_col].values.astype(np.int32)

        # Normalize features for LSTM
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i])
            y.append(targets[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        return X, y, feature_cols

    # ===========================================================
    # PRIVATE: Indicator computation methods
    # ===========================================================

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features."""
        # Returns over different periods
        for period in self.config.lookback_periods:
            df[f"return_{period}"] = df["close"].pct_change(period)
            df[f"log_return_{period}"] = np.log(df["close"] / df["close"].shift(period))

        # High-low range
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]

        # Close position within range
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # Gap (open vs previous close)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Body size (open-close)
        df["body_size"] = abs(df["close"] - df["open"]) / df["close"]

        # Upper/lower shadow
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple and Exponential Moving Averages."""
        for period in self.config.sma_periods:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
            # Distance from SMA
            df[f"dist_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

        for period in self.config.ema_periods:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"dist_ema_{period}"] = (df["close"] - df[f"ema_{period}"]) / df[f"ema_{period}"]

        # MA crossover signals
        if len(self.config.sma_periods) >= 2:
            fast = self.config.sma_periods[0]
            slow = self.config.sma_periods[-2]
            df["ma_cross"] = (df[f"sma_{fast}"] - df[f"sma_{slow}"]) / df[f"sma_{slow}"]

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index."""
        period = self.config.rsi_period
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI zones
        df["rsi_overbought"] = (df["rsi"] > self.config.rsi_overbought).astype(int)
        df["rsi_oversold"] = (df["rsi"] < self.config.rsi_oversold).astype(int)

        # RSI slope
        df["rsi_slope"] = df["rsi"].diff(3)

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD indicator."""
        ema_fast = df["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.config.macd_slow, adjust=False).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=self.config.macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # MACD crossover
        df["macd_cross"] = np.sign(df["macd_histogram"]) - np.sign(df["macd_histogram"].shift(1))

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands."""
        period = self.config.bb_period
        std_mult = self.config.bb_std

        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()

        df["bb_upper"] = sma + std_mult * std
        df["bb_lower"] = sma - std_mult * std
        df["bb_middle"] = sma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range."""
        period = self.config.atr_period

        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=period).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic Oscillator."""
        k_period = self.config.stoch_k
        d_period = self.config.stoch_d

        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
        df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average Directional Index."""
        period = self.config.adx_period

        # +DM and -DM
        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # True Range
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed - pass df.index to avoid integer vs datetime index misalignment
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / (atr + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.rolling(window=period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        df["di_diff"] = plus_di - minus_di

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        if "volume" not in df.columns:
            return df

        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
        df["volume_change"] = df["volume"].pct_change()

        # On-Balance Volume (OBV) trend
        obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["obv_slope"] = obv.diff(5)

        return df

    def _add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple candlestick pattern features."""
        # Bullish / Bearish candle
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)

        # Doji (very small body)
        body = abs(df["close"] - df["open"])
        hl_range = df["high"] - df["low"]
        df["is_doji"] = (body / (hl_range + 1e-10) < 0.1).astype(int)

        # Engulfing
        prev_body = abs(df["close"].shift(1) - df["open"].shift(1))
        df["is_engulfing"] = (body > prev_body * 1.5).astype(int)

        # Consecutive direction
        df["consec_bullish"] = df["is_bullish"].rolling(window=3).sum()

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        # Session indicators
        df["london_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
        df["ny_session"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)
        df["tokyo_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
        df["overlap_session"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators."""
        # Rate of change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / df["close"].shift(period)

        # Williams %R
        period = 14
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()
        df["williams_r"] = -100 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-10)

        # CCI (Commodity Channel Index)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df["cci"] = (tp - sma_tp) / (0.015 * mad + 1e-10)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility features."""
        # Rolling standard deviation of returns
        returns = df["close"].pct_change()
        for period in [5, 10, 20]:
            df[f"volatility_{period}"] = returns.rolling(window=period).std()

        # Parkinson volatility (using high-low)
        df["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) *
            (np.log(df["high"] / df["low"]) ** 2).rolling(window=20).mean()
        )

        # Volatility ratio (short vs long)
        if "volatility_5" in df.columns and "volatility_20" in df.columns:
            df["vol_ratio"] = df["volatility_5"] / (df["volatility_20"] + 1e-10)

        return df
