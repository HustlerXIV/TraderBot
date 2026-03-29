"""
MetaTrader 5 Connection Module
Handles all communication with the MT5 terminal:
- Connecting/disconnecting
- Fetching historical data
- Placing/modifying/closing orders
- Account information
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .config import BotConfig, TIMEFRAMES

logger = logging.getLogger(__name__)


# MT5 timeframe mapping
MT5_TIMEFRAMES = {}
if MT5_AVAILABLE:
    MT5_TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }


class MT5Connector:
    """Manages connection and operations with MetaTrader 5."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.connected = False
        self._check_mt5()

    def _check_mt5(self):
        """Check if MT5 library is available."""
        if not MT5_AVAILABLE:
            logger.warning(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5 "
                "(Windows only). Running in simulation mode."
            )

    def connect(self) -> bool:
        """Initialize and connect to MT5 terminal."""
        if not MT5_AVAILABLE:
            logger.info("MT5 not available - using simulation mode")
            self.connected = False
            return False

        try:
            # Initialize MT5
            init_params = {}
            if self.config.mt5.path:
                init_params["path"] = self.config.mt5.path

            if not mt5.initialize(**init_params):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

            # Login if credentials provided
            if self.config.mt5.login and self.config.mt5.password:
                authorized = mt5.login(
                    login=self.config.mt5.login,
                    password=self.config.mt5.password,
                    server=self.config.mt5.server,
                    timeout=self.config.mt5.timeout
                )
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False

            self.connected = True
            account_info = mt5.account_info()
            logger.info(
                f"Connected to MT5 | Account: {account_info.login} | "
                f"Balance: {account_info.balance} {account_info.currency} | "
                f"Server: {account_info.server}"
            )
            return True

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from MT5."""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_account_info(self) -> Dict:
        """Get current account information."""
        if not self.connected:
            return self._simulated_account_info()

        info = mt5.account_info()
        if info is None:
            return {}

        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
            "server": info.server,
        }

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        bars: int = 5000,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from MT5.

        Args:
            symbol: Currency pair (e.g. "EURUSD")
            timeframe: Timeframe string (e.g. "H1", "D1")
            bars: Number of bars to fetch
            start_date: Optional start date

        Returns:
            DataFrame with columns: time, open, high, low, close, volume, spread
        """
        if not self.connected:
            return self._generate_simulated_data(symbol, timeframe, bars)

        tf = MT5_TIMEFRAMES.get(timeframe)
        if tf is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        try:
            if start_date:
                rates = mt5.copy_rates_from(symbol, tf, start_date, bars)
            else:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.rename(columns={
                "tick_volume": "volume"
            }, inplace=True)

            # Keep only needed columns
            cols = ["open", "high", "low", "close", "volume", "spread"]
            available = [c for c in cols if c in df.columns]
            df = df[available]

            logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading information (pip value, lot size, etc.)."""
        if not self.connected:
            return self._simulated_symbol_info(symbol)

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol info not found: {symbol}")
            return None

        return {
            "symbol": symbol,
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "bid": info.bid,
            "ask": info.ask,
        }

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask price for a symbol."""
        if not self.connected:
            return {"bid": 1.1000, "ask": 1.1002, "spread": 2}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round((tick.ask - tick.bid) / mt5.symbol_info(symbol).point),
            "time": datetime.fromtimestamp(tick.time),
        }

    def place_order(
        self,
        symbol: str,
        order_type: str,  # "buy" or "sell"
        lot_size: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "AI_Bot",
        magic: int = 123456
    ) -> Optional[Dict]:
        """
        Place a market order.

        Args:
            symbol: Currency pair
            order_type: "buy" or "sell"
            lot_size: Volume in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            magic: Magic number for identification

        Returns:
            Order result dict or None
        """
        if self.config.paper_trading:
            return self._paper_trade_order(
                symbol, order_type, lot_size, stop_loss, take_profit, comment
            )

        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        price_info = self.get_current_price(symbol)
        if not price_info:
            return None

        if order_type.lower() == "buy":
            trade_type = mt5.ORDER_TYPE_BUY
            price = price_info["ask"]
        elif order_type.lower() == "sell":
            trade_type = mt5.ORDER_TYPE_SELL
            price = price_info["bid"]
        else:
            logger.error(f"Invalid order type: {order_type}")
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": trade_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        logger.info(
            f"Order placed: {order_type.upper()} {lot_size} {symbol} "
            f"@ {price} | SL: {stop_loss} | TP: {take_profit} | "
            f"Ticket: {result.order}"
        )

        return {
            "ticket": result.order,
            "symbol": symbol,
            "type": order_type,
            "lot_size": lot_size,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def close_position(self, ticket: int) -> bool:
        """Close a specific position by ticket number."""
        if self.config.paper_trading:
            logger.info(f"[PAPER] Closed position #{ticket}")
            return True

        if not self.connected:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} not found")
            return False

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "AI_Bot_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} closed @ {price}")
            return True

        logger.error(f"Failed to close position {ticket}")
        return False

    def modify_position(
        self, ticket: int, stop_loss: float = None, take_profit: float = None
    ) -> bool:
        """Modify SL/TP of an open position."""
        if self.config.paper_trading:
            logger.info(f"[PAPER] Modified position #{ticket} | SL: {stop_loss} | TP: {take_profit}")
            return True

        if not self.connected:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": stop_loss if stop_loss else pos.sl,
            "tp": take_profit if take_profit else pos.tp,
        }

        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self.connected:
            return []

        positions = mt5.positions_get()
        if not positions:
            return []

        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "buy" if p.type == 0 else "sell",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "stop_loss": p.sl,
                "take_profit": p.tp,
                "profit": p.profit,
                "swap": p.swap,
                "magic": p.magic,
                "comment": p.comment,
                "time": datetime.fromtimestamp(p.time),
            }
            for p in positions
        ]

    # ---- Simulation helpers (for when MT5 is not available) ----

    def _simulated_account_info(self) -> Dict:
        return {
            "login": 0,
            "balance": 10000.0,
            "equity": 10000.0,
            "margin": 0.0,
            "free_margin": 10000.0,
            "margin_level": 0.0,
            "profit": 0.0,
            "currency": "USD",
            "leverage": 100,
            "server": "Simulation",
        }

    def _generate_simulated_data(
        self, symbol: str, timeframe: str, bars: int
    ) -> pd.DataFrame:
        """Generate realistic-looking simulated OHLCV data for testing."""
        logger.info(f"Generating simulated data for {symbol} {timeframe} ({bars} bars)")

        np.random.seed(hash(symbol) % 2**31)
        minutes = TIMEFRAMES.get(timeframe, 60)
        end = datetime.now()
        start = end - timedelta(minutes=minutes * bars)
        dates = pd.date_range(start=start, periods=bars, freq=f"{minutes}min")

        # Base prices for different pairs
        base_prices = {
            "EURUSD": 1.1000, "GBPUSD": 1.2700, "USDJPY": 145.00,
            "USDCHF": 0.8800, "AUDUSD": 0.6500, "USDCAD": 1.3600,
            "NZDUSD": 0.6100, "USDTRY": 32.00, "USDZAR": 18.00,
        }
        base = base_prices.get(symbol, 1.0000)

        # Random walk with drift
        returns = np.random.normal(0, 0.0005, bars)
        prices = base * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = []
        for i, price in enumerate(prices):
            volatility = abs(np.random.normal(0, 0.001))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_p = price + np.random.normal(0, 0.0002)
            close = price + np.random.normal(0, 0.0002)
            volume = int(np.random.exponential(1000))
            data.append([open_p, max(high, open_p, close), min(low, open_p, close), close, volume, 2])

        df = pd.DataFrame(
            data, index=dates,
            columns=["open", "high", "low", "close", "volume", "spread"]
        )
        df.index.name = "time"
        return df

    def _simulated_symbol_info(self, symbol: str) -> Dict:
        digits = 5 if "JPY" not in symbol else 3
        point = 0.00001 if "JPY" not in symbol else 0.001
        return {
            "symbol": symbol,
            "point": point,
            "digits": digits,
            "spread": 15,
            "trade_contract_size": 100000,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "bid": 1.1000,
            "ask": 1.1002,
        }

    def _paper_trade_order(self, symbol, order_type, lot_size, sl, tp, comment) -> Dict:
        """Simulate order execution for paper trading."""
        price = 1.1001 if order_type == "buy" else 1.0999
        ticket = int(datetime.now().timestamp() * 1000) % 10**8

        logger.info(
            f"[PAPER TRADE] {order_type.upper()} {lot_size} {symbol} "
            f"@ {price} | SL: {sl} | TP: {tp}"
        )

        return {
            "ticket": ticket,
            "symbol": symbol,
            "type": order_type,
            "lot_size": lot_size,
            "price": price,
            "stop_loss": sl,
            "take_profit": tp,
        }
