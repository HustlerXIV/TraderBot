"""
OANDA REST API Connector
Works natively on Mac/Linux/Windows - no MT5 needed.

Supports:
- Live and Practice (demo) accounts
- Historical candlestick data
- Market orders, limit orders
- Position management
- Account info and streaming prices

Setup:
1. Create free practice account at https://www.oanda.com/
2. Generate API token in your OANDA account settings
3. Note your account ID (starts with digits, e.g., "101-001-12345678-001")
"""

import logging
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .config import BotConfig, TIMEFRAMES

logger = logging.getLogger(__name__)


# OANDA API endpoints
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"

# OANDA timeframe mapping
OANDA_TIMEFRAMES = {
    "M1": "M1",
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "D1": "D",
}

# OANDA uses underscore format for pairs (EUR_USD instead of EURUSD)
def to_oanda_symbol(symbol: str) -> str:
    """Convert EURUSD -> EUR_USD."""
    if "_" in symbol:
        return symbol
    return symbol[:3] + "_" + symbol[3:]

def from_oanda_symbol(symbol: str) -> str:
    """Convert EUR_USD -> EURUSD."""
    return symbol.replace("_", "")


class OandaConnector:
    """
    OANDA REST API connector for Forex trading.
    Drop-in replacement for MT5Connector with the same interface.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.connected = False

        # OANDA credentials (from config)
        self.api_token = config.oanda.api_token
        self.account_id = config.oanda.account_id
        self.is_practice = config.oanda.practice

        # Set base URL
        self.base_url = OANDA_PRACTICE_URL if self.is_practice else OANDA_LIVE_URL

        # Session for connection pooling
        self.session = None

        self._check_requests()

    def _check_requests(self):
        if not REQUESTS_AVAILABLE:
            logger.error(
                "requests package not installed. "
                "Install with: pip install requests"
            )

    def _headers(self) -> Dict:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        data: Dict = None
    ) -> Optional[Dict]:
        """Make an API request to OANDA."""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=data,
                timeout=30
            )

            if response.status_code in [200, 201]:
                return response.json()
            else:
                error_body = response.text
                logger.error(
                    f"OANDA API error {response.status_code}: {error_body}"
                )
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    # ====================================================
    # CONNECTION
    # ====================================================

    def connect(self) -> bool:
        """Connect to OANDA API and verify credentials."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Running in simulation mode (requests not installed)")
            self.connected = False
            return False

        if not self.api_token or not self.account_id:
            logger.warning(
                "OANDA credentials not configured. "
                "Set api_token and account_id in config. "
                "Running in simulation mode."
            )
            self.connected = False
            return False

        self.session = requests.Session()

        # Test connection by fetching account info
        result = self._request("GET", f"/v3/accounts/{self.account_id}")
        if result and "account" in result:
            acc = result["account"]
            logger.info(
                f"Connected to OANDA | Account: {self.account_id} | "
                f"Balance: {acc.get('balance', 'N/A')} {acc.get('currency', 'USD')} | "
                f"Mode: {'Practice' if self.is_practice else 'LIVE'}"
            )
            self.connected = True
            return True
        else:
            logger.error("Failed to connect to OANDA. Check your API token and account ID.")
            self.connected = False
            return False

    def disconnect(self):
        """Close the OANDA session."""
        if self.session:
            self.session.close()
        self.connected = False
        logger.info("Disconnected from OANDA")

    # ====================================================
    # ACCOUNT INFO
    # ====================================================

    def get_account_info(self) -> Dict:
        """Get current account information."""
        if not self.connected:
            return self._simulated_account_info()

        result = self._request("GET", f"/v3/accounts/{self.account_id}/summary")
        if not result or "account" not in result:
            return self._simulated_account_info()

        acc = result["account"]
        return {
            "login": self.account_id,
            "balance": float(acc.get("balance", 0)),
            "equity": float(acc.get("NAV", 0)),
            "margin": float(acc.get("marginUsed", 0)),
            "free_margin": float(acc.get("marginAvailable", 0)),
            "margin_level": float(acc.get("marginRate", 0)) * 100,
            "profit": float(acc.get("unrealizedPL", 0)),
            "currency": acc.get("currency", "USD"),
            "leverage": int(1 / float(acc.get("marginRate", 0.01))),
            "server": "OANDA Practice" if self.is_practice else "OANDA Live",
        }

    # ====================================================
    # HISTORICAL DATA
    # ====================================================

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        bars: int = 5000,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV candles from OANDA.
        OANDA limits to 5000 candles per request.
        """
        if not self.connected:
            return self._generate_simulated_data(symbol, timeframe, bars)

        oanda_symbol = to_oanda_symbol(symbol)
        oanda_tf = OANDA_TIMEFRAMES.get(timeframe, "H1")

        all_candles = []
        remaining = bars
        end_time = None

        # Fetch in chunks (OANDA max 5000 per request)
        while remaining > 0:
            chunk_size = min(remaining, 5000)
            params = {
                "granularity": oanda_tf,
                "count": chunk_size,
                "price": "MBA",  # Mid, Bid, Ask
            }

            if end_time:
                params["to"] = end_time

            result = self._request(
                "GET",
                f"/v3/instruments/{oanda_symbol}/candles",
                params=params
            )

            if not result or "candles" not in result:
                break

            candles = result["candles"]
            if not candles:
                break

            all_candles = candles + all_candles
            remaining -= len(candles)

            # Set end_time to the earliest candle for next chunk
            end_time = candles[0]["time"]

            # Avoid hitting rate limits
            time.sleep(0.1)

        if not all_candles:
            logger.warning(f"No data received for {symbol} {timeframe}")
            return self._generate_simulated_data(symbol, timeframe, bars)

        # Parse into DataFrame
        data = []
        for candle in all_candles:
            if not candle.get("complete", True):
                continue  # Skip incomplete candles

            mid = candle.get("mid", {})
            bid = candle.get("bid", {})
            ask = candle.get("ask", {})

            data.append({
                "time": pd.to_datetime(candle["time"]),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(candle.get("volume", 0)),
                "spread": round(
                    (float(ask.get("c", 0)) - float(bid.get("c", 0)))
                    / self._get_pip_size(symbol) if bid.get("c") else 0,
                    1
                ),
            })

        df = pd.DataFrame(data)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe} from OANDA")
        return df

    # ====================================================
    # SYMBOL / PRICE INFO
    # ====================================================

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading information."""
        if not self.connected:
            return self._simulated_symbol_info(symbol)

        oanda_symbol = to_oanda_symbol(symbol)
        result = self._request("GET", f"/v3/accounts/{self.account_id}/instruments",
                               params={"instruments": oanda_symbol})

        if not result or "instruments" not in result:
            return self._simulated_symbol_info(symbol)

        instr = result["instruments"][0]
        pip_location = int(instr.get("pipLocation", -4))
        pip_size = 10 ** pip_location
        display_precision = int(instr.get("displayPrecision", 5))

        return {
            "symbol": symbol,
            "point": 10 ** (-display_precision),
            "digits": display_precision,
            "spread": 0,  # Dynamic, from pricing
            "trade_contract_size": int(instr.get("maximumOrderUnits", 100000)),
            "volume_min": float(instr.get("minimumTradeSize", 1)),
            "volume_max": float(instr.get("maximumOrderUnits", 100000000)),
            "volume_step": 1,  # OANDA uses units, not lots
            "pip_size": pip_size,
            "bid": 0,
            "ask": 0,
        }

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask price."""
        if not self.connected:
            return {"bid": 1.1000, "ask": 1.1002, "spread": 2}

        oanda_symbol = to_oanda_symbol(symbol)
        result = self._request(
            "GET",
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": oanda_symbol}
        )

        if not result or "prices" not in result or not result["prices"]:
            return None

        price = result["prices"][0]
        bid = float(price["bids"][0]["price"]) if price.get("bids") else 0
        ask = float(price["asks"][0]["price"]) if price.get("asks") else 0
        pip_size = self._get_pip_size(symbol)

        return {
            "bid": bid,
            "ask": ask,
            "spread": round((ask - bid) / pip_size, 1) if pip_size else 0,
            "time": datetime.now(timezone.utc),
        }

    # ====================================================
    # ORDER EXECUTION
    # ====================================================

    def place_order(
        self,
        symbol: str,
        order_type: str,      # "buy" or "sell"
        lot_size: float,      # In lots (will convert to OANDA units)
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "AI_Bot",
        magic: int = 123456
    ) -> Optional[Dict]:
        """Place a market order via OANDA."""
        if self.config.paper_trading:
            return self._paper_trade_order(symbol, order_type, lot_size, stop_loss, take_profit, comment)

        if not self.connected:
            logger.error("Not connected to OANDA")
            return None

        oanda_symbol = to_oanda_symbol(symbol)

        # Convert lots to OANDA units (1 lot = 100,000 units)
        units = int(lot_size * 100000)
        if order_type.lower() == "sell":
            units = -units

        # Build order body
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": oanda_symbol,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
                "positionFill": "DEFAULT",
            }
        }

        # Add stop loss
        if stop_loss > 0:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}",
                "timeInForce": "GTC"
            }

        # Add take profit
        if take_profit > 0:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}",
                "timeInForce": "GTC"
            }

        # Add comment/tag
        order_data["order"]["clientExtensions"] = {
            "comment": comment,
            "tag": str(magic)
        }

        result = self._request(
            "POST",
            f"/v3/accounts/{self.account_id}/orders",
            data=order_data
        )

        if not result:
            logger.error("Order placement failed")
            return None

        # Check if order was filled
        if "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            trade_id = fill.get("tradeOpened", {}).get("tradeID", "0")
            price = float(fill.get("price", 0))

            logger.info(
                f"Order filled: {order_type.upper()} {lot_size} lots {symbol} "
                f"@ {price} | SL: {stop_loss} | TP: {take_profit} | "
                f"Trade ID: {trade_id}"
            )

            return {
                "ticket": int(trade_id),
                "symbol": symbol,
                "type": order_type,
                "lot_size": lot_size,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

        elif "orderCancelTransaction" in result:
            cancel = result["orderCancelTransaction"]
            logger.error(f"Order cancelled: {cancel.get('reason', 'Unknown')}")
            return None

        logger.error(f"Unexpected order response: {result}")
        return None

    def close_position(self, ticket: int) -> bool:
        """Close a specific trade by its ID."""
        if self.config.paper_trading:
            logger.info(f"[PAPER] Closed trade #{ticket}")
            return True

        if not self.connected:
            return False

        result = self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/trades/{ticket}/close"
        )

        if result and "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            logger.info(f"Trade {ticket} closed @ {fill.get('price')}")
            return True

        logger.error(f"Failed to close trade {ticket}")
        return False

    def modify_position(
        self,
        ticket: int,
        stop_loss: float = None,
        take_profit: float = None
    ) -> bool:
        """Modify SL/TP of an open trade."""
        if self.config.paper_trading:
            logger.info(f"[PAPER] Modified trade #{ticket} | SL: {stop_loss} | TP: {take_profit}")
            return True

        if not self.connected:
            return False

        data = {}
        if stop_loss is not None:
            data["stopLoss"] = {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"}
        if take_profit is not None:
            data["takeProfit"] = {"price": f"{take_profit:.5f}", "timeInForce": "GTC"}

        result = self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/trades/{ticket}/orders",
            data=data
        )

        if result:
            logger.info(f"Trade {ticket} modified | SL: {stop_loss} | TP: {take_profit}")
            return True
        return False

    def get_open_positions(self) -> List[Dict]:
        """Get all open trades."""
        if not self.connected:
            return []

        result = self._request("GET", f"/v3/accounts/{self.account_id}/openTrades")
        if not result or "trades" not in result:
            return []

        positions = []
        for trade in result["trades"]:
            units = int(trade.get("currentUnits", 0))
            positions.append({
                "ticket": int(trade["id"]),
                "symbol": from_oanda_symbol(trade["instrument"]),
                "type": "buy" if units > 0 else "sell",
                "volume": abs(units) / 100000,  # Convert units to lots
                "open_price": float(trade.get("price", 0)),
                "current_price": float(trade.get("unrealizedPL", 0)),  # Approx
                "stop_loss": float(trade.get("stopLossOrder", {}).get("price", 0)) if trade.get("stopLossOrder") else 0,
                "take_profit": float(trade.get("takeProfitOrder", {}).get("price", 0)) if trade.get("takeProfitOrder") else 0,
                "profit": float(trade.get("unrealizedPL", 0)),
                "swap": float(trade.get("financing", 0)),
                "magic": trade.get("clientExtensions", {}).get("tag", "0"),
                "comment": trade.get("clientExtensions", {}).get("comment", ""),
                "time": pd.to_datetime(trade.get("openTime")),
            })

        return positions

    # ====================================================
    # HELPERS
    # ====================================================

    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for a symbol."""
        if "JPY" in symbol.upper():
            return 0.01
        return 0.0001

    # ====================================================
    # SIMULATION (same interface as MT5Connector)
    # ====================================================

    def _simulated_account_info(self) -> Dict:
        return {
            "login": "SIMULATION",
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
        """Generate realistic simulated OHLCV data."""
        logger.info(f"Generating simulated data for {symbol} {timeframe} ({bars} bars)")

        np.random.seed(hash(symbol) % 2**31)
        minutes = TIMEFRAMES.get(timeframe, 60)
        end = datetime.now()
        start = end - timedelta(minutes=minutes * bars)
        dates = pd.date_range(start=start, periods=bars, freq=f"{minutes}min")

        base_prices = {
            "EURUSD": 1.1000, "GBPUSD": 1.2700, "USDJPY": 145.00,
            "USDCHF": 0.8800, "AUDUSD": 0.6500, "USDCAD": 1.3600,
            "NZDUSD": 0.6100, "USDTRY": 32.00, "USDZAR": 18.00,
        }
        base = base_prices.get(symbol, 1.0000)

        returns = np.random.normal(0, 0.0005, bars)
        prices = base * np.exp(np.cumsum(returns))

        data = []
        for i, price in enumerate(prices):
            volatility = abs(np.random.normal(0, 0.001))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_p = price + np.random.normal(0, 0.0002)
            close = price + np.random.normal(0, 0.0002)
            volume = int(np.random.exponential(1000))
            data.append([open_p, max(high, open_p, close), min(low, open_p, close), close, volume, 2])

        df = pd.DataFrame(data, index=dates, columns=["open", "high", "low", "close", "volume", "spread"])
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
            "pip_size": 0.0001 if "JPY" not in symbol else 0.01,
            "bid": 1.1000,
            "ask": 1.1002,
        }

    def _paper_trade_order(self, symbol, order_type, lot_size, sl, tp, comment) -> Dict:
        """Simulate order for paper trading."""
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
