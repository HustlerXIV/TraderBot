"""
Risk Management System
Controls position sizing, stop losses, max exposure,
drawdown protection, and trade frequency limits.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import deque

from .config import RiskConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Comprehensive risk management for the trading bot.

    Enforces:
    - Maximum risk per trade (% of balance)
    - Maximum total exposure
    - Max open positions
    - Daily trade limits
    - Consecutive loss protection
    - Maximum drawdown circuit breaker
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.trade_log: List[Dict] = []
        self.daily_trades = 0
        self.daily_reset_date = datetime.now().date()
        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None
        self.peak_balance = 0.0
        self.is_halted = False
        self.halt_reason = ""

    def can_trade(self, account_info: Dict, open_positions: List[Dict]) -> tuple:
        """
        Check if a new trade is allowed.

        Returns:
            (allowed: bool, reason: str)
        """
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self.daily_reset_date:
            self.daily_trades = 0
            self.daily_reset_date = today

        # Check if halted
        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # Check cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds // 60
            return False, f"Cooling down after consecutive losses ({remaining} min remaining)"

        # Check max open trades
        if len(open_positions) >= self.config.max_open_trades:
            return False, f"Max open trades reached ({self.config.max_open_trades})"

        # Check daily trade limit
        if self.daily_trades >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.config.max_trades_per_day})"

        # Check total exposure
        balance = account_info.get("balance", 0)
        if balance <= 0:
            return False, "Invalid balance"

        total_exposure = sum(abs(p.get("volume", 0)) for p in open_positions)
        max_lots = balance / 100000 * self.config.max_total_exposure / 100 * account_info.get("leverage", 100)
        if total_exposure >= max_lots:
            return False, f"Max total exposure reached"

        # Check drawdown
        equity = account_info.get("equity", balance)
        if self.peak_balance == 0:
            self.peak_balance = balance
        self.peak_balance = max(self.peak_balance, balance)

        drawdown = (self.peak_balance - equity) / self.peak_balance * 100
        if drawdown >= self.config.max_drawdown_percent:
            self.is_halted = True
            self.halt_reason = f"Max drawdown reached ({drawdown:.1f}%)"
            return False, self.halt_reason

        return True, "OK"

    def calculate_position_size(
        self,
        account_info: Dict,
        stop_loss_pips: float,
        symbol_info: Dict
    ) -> float:
        """
        Calculate optimal position size based on risk parameters.

        Uses fixed fractional position sizing:
        Lot Size = (Balance * Risk%) / (SL in pips * Pip Value)

        Returns:
            Position size in lots
        """
        balance = account_info.get("balance", 10000)
        risk_amount = balance * (self.config.max_risk_per_trade / 100)

        # Pip value calculation
        contract_size = symbol_info.get("trade_contract_size", 100000)
        point = symbol_info.get("point", 0.00001)
        digits = symbol_info.get("digits", 5)

        # For most pairs, 1 pip = 10 points for 5-digit brokers
        pip_size = point * 10 if digits == 5 or digits == 3 else point

        # Pip value per standard lot
        pip_value = pip_size * contract_size

        if stop_loss_pips <= 0:
            logger.warning("Invalid stop loss, using minimum lot size")
            return self.config.min_lot_size

        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value)

        # Respect volume constraints
        vol_step = symbol_info.get("volume_step", 0.01)
        lot_size = round(lot_size / vol_step) * vol_step

        # Clamp to min/max
        lot_size = max(self.config.min_lot_size, min(self.config.max_lot_size, lot_size))

        logger.info(
            f"Position sizing: Balance={balance}, Risk={risk_amount:.2f}, "
            f"SL={stop_loss_pips} pips, Lot={lot_size}"
        )
        return lot_size

    def calculate_stop_loss(
        self,
        signal: int,
        current_price: float,
        atr: float,
        symbol_info: Dict
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            signal: 1=BUY, -1=SELL
            current_price: Current price
            atr: Current ATR value
            symbol_info: Symbol info dict

        Returns:
            Stop loss price
        """
        sl_distance = atr * self.config.stop_loss_atr_multiplier
        digits = symbol_info.get("digits", 5)

        if signal == 1:  # BUY - SL below price
            sl = current_price - sl_distance
        elif signal == -1:  # SELL - SL above price
            sl = current_price + sl_distance
        else:
            return 0.0

        return round(sl, digits)

    def calculate_take_profit(
        self,
        signal: int,
        current_price: float,
        atr: float,
        symbol_info: Dict
    ) -> float:
        """
        Calculate take profit price based on ATR.

        Returns:
            Take profit price
        """
        tp_distance = atr * self.config.take_profit_atr_multiplier
        digits = symbol_info.get("digits", 5)

        if signal == 1:  # BUY - TP above price
            tp = current_price + tp_distance
        elif signal == -1:  # SELL - TP below price
            tp = current_price - tp_distance
        else:
            return 0.0

        return round(tp, digits)

    def should_trail_stop(
        self,
        position: Dict,
        current_price: float,
        symbol_info: Dict
    ) -> Optional[float]:
        """
        Check if trailing stop should be updated.

        Returns:
            New stop loss price if should update, None otherwise
        """
        point = symbol_info.get("point", 0.00001)
        digits = symbol_info.get("digits", 5)
        activation_pips = self.config.trailing_stop_activation
        trail_pips = self.config.trailing_stop_distance

        open_price = position.get("open_price", 0)
        current_sl = position.get("stop_loss", 0)
        pos_type = position.get("type", "buy")

        if pos_type == "buy":
            profit_pips = (current_price - open_price) / (point * 10)
            if profit_pips >= activation_pips:
                new_sl = current_price - trail_pips * point * 10
                new_sl = round(new_sl, digits)
                if new_sl > current_sl:
                    return new_sl
        elif pos_type == "sell":
            profit_pips = (open_price - current_price) / (point * 10)
            if profit_pips >= activation_pips:
                new_sl = current_price + trail_pips * point * 10
                new_sl = round(new_sl, digits)
                if current_sl == 0 or new_sl < current_sl:
                    return new_sl

        return None

    def record_trade_result(self, profit: float):
        """Record a completed trade result for tracking."""
        self.trade_log.append({
            "time": datetime.now().isoformat(),
            "profit": profit
        })
        self.daily_trades += 1

        if profit < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                self.cooldown_until = datetime.now() + timedelta(
                    minutes=self.config.loss_cooldown_minutes
                )
                logger.warning(
                    f"Consecutive loss limit reached ({self.consecutive_losses}). "
                    f"Cooling down for {self.config.loss_cooldown_minutes} minutes."
                )
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

    def get_stats(self) -> Dict:
        """Get risk management statistics."""
        if not self.trade_log:
            return {"total_trades": 0}

        profits = [t["profit"] for t in self.trade_log]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        return {
            "total_trades": len(profits),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(profits) * 100 if profits else 0,
            "total_profit": sum(profits),
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "consecutive_losses": self.consecutive_losses,
            "is_halted": self.is_halted,
            "daily_trades": self.daily_trades,
        }

    def resume_trading(self):
        """Resume trading after halt."""
        self.is_halted = False
        self.halt_reason = ""
        self.cooldown_until = None
        self.consecutive_losses = 0
        logger.info("Trading resumed")
