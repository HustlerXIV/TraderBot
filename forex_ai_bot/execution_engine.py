"""
Trade Execution Engine
Handles order placement, position management,
trailing stops, and trade lifecycle.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List

from .config import BotConfig
from .mt5_connector import MT5Connector
from .risk_manager import RiskManager
from .strategy import TradingSignal

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Manages trade execution and position lifecycle.

    Responsibilities:
    - Execute trading signals as market orders
    - Monitor and manage open positions
    - Apply trailing stops
    - Handle partial closes
    - Track all trade history
    """

    def __init__(
        self,
        config: BotConfig,
        connector: MT5Connector,
        risk_manager: RiskManager
    ):
        self.config = config
        self.connector = connector
        self.risk_manager = risk_manager
        self.active_trades: Dict[int, Dict] = {}  # ticket -> trade info
        self.trade_history: List[Dict] = []

    def execute_signal(self, signal: TradingSignal) -> Optional[Dict]:
        """
        Execute a trading signal by placing an order.

        Args:
            signal: TradingSignal object

        Returns:
            Trade result dict or None if rejected
        """
        # Pre-execution checks
        account_info = self.connector.get_account_info()
        open_positions = self.connector.get_open_positions()

        can_trade, reason = self.risk_manager.can_trade(account_info, open_positions)
        if not can_trade:
            logger.warning(f"Trade rejected: {reason}")
            return None

        # Check if we already have a position on this pair
        existing = [p for p in open_positions if p["symbol"] == signal.symbol]
        if existing:
            logger.info(f"Already have position on {signal.symbol}, skipping")
            return None

        # Place order
        order_type = "buy" if signal.signal == 1 else "sell"
        result = self.connector.place_order(
            symbol=signal.symbol,
            order_type=order_type,
            lot_size=signal.lot_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            comment=f"AI_{signal.confidence:.0%}"
        )

        if result:
            trade_info = {
                **result,
                "signal_confidence": signal.confidence,
                "signal_details": signal.details,
                "open_time": datetime.now().isoformat(),
                "timeframe": signal.timeframe,
            }
            self.active_trades[result["ticket"]] = trade_info
            logger.info(f"Trade executed: {trade_info}")
            return trade_info

        return None

    def manage_positions(self):
        """
        Monitor and manage all open positions.

        - Update trailing stops
        - Check for manual close conditions
        - Track position health
        """
        positions = self.connector.get_open_positions()

        for position in positions:
            ticket = position["ticket"]
            symbol = position["symbol"]

            try:
                # Get current price and symbol info
                symbol_info = self.connector.get_symbol_info(symbol)
                price_info = self.connector.get_current_price(symbol)

                if not symbol_info or not price_info:
                    continue

                current_price = price_info["bid"] if position["type"] == "buy" else price_info["ask"]

                # Check trailing stop
                new_sl = self.risk_manager.should_trail_stop(
                    position, current_price, symbol_info
                )
                if new_sl is not None:
                    success = self.connector.modify_position(
                        ticket, stop_loss=new_sl
                    )
                    if success:
                        logger.info(
                            f"Trailing stop updated: {symbol} #{ticket} "
                            f"SL -> {new_sl}"
                        )

            except Exception as e:
                logger.error(f"Error managing position {ticket}: {e}")

    def close_all_positions(self, reason: str = "Manual close"):
        """Close all open positions."""
        positions = self.connector.get_open_positions()
        for position in positions:
            ticket = position["ticket"]
            try:
                self.connector.close_position(ticket)
                self._record_closed_trade(position, reason)
                logger.info(f"Closed position #{ticket} ({reason})")
            except Exception as e:
                logger.error(f"Failed to close #{ticket}: {e}")

    def close_position_by_symbol(self, symbol: str, reason: str = "Signal reversal"):
        """Close all positions for a specific symbol."""
        positions = self.connector.get_open_positions()
        for position in positions:
            if position["symbol"] == symbol:
                try:
                    self.connector.close_position(position["ticket"])
                    self._record_closed_trade(position, reason)
                except Exception as e:
                    logger.error(f"Failed to close {symbol}: {e}")

    def check_closed_trades(self):
        """
        Check if any tracked trades have been closed (by SL/TP)
        and record the results.
        """
        current_positions = self.connector.get_open_positions()
        current_tickets = {p["ticket"] for p in current_positions}

        closed_tickets = []
        for ticket in list(self.active_trades.keys()):
            if ticket not in current_tickets:
                trade = self.active_trades[ticket]
                self._record_closed_trade(trade, "SL/TP hit")
                closed_tickets.append(ticket)

        for ticket in closed_tickets:
            del self.active_trades[ticket]

    def _record_closed_trade(self, trade: Dict, reason: str):
        """Record a closed trade in history and risk manager."""
        profit = trade.get("profit", 0)
        self.risk_manager.record_trade_result(profit)

        record = {
            **trade,
            "close_time": datetime.now().isoformat(),
            "close_reason": reason,
        }
        self.trade_history.append(record)

        logger.info(
            f"Trade closed: {trade.get('symbol')} | "
            f"Profit: {profit} | Reason: {reason}"
        )

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        positions = self.connector.get_open_positions()
        account = self.connector.get_account_info()

        total_profit = sum(p.get("profit", 0) for p in positions)
        symbols = [p["symbol"] for p in positions]

        return {
            "account": account,
            "open_positions": len(positions),
            "symbols": symbols,
            "unrealized_pnl": total_profit,
            "total_trades": len(self.trade_history),
            "risk_stats": self.risk_manager.get_stats(),
        }
