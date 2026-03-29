"""
Entry point for the Forex AI Trading Bot.

Usage:
    python3 -m forex_ai_bot                          # Start paper trading (OANDA by default)
    python3 -m forex_ai_bot --backtest               # Run backtest
    python3 -m forex_ai_bot --train                  # Train models only
    python3 -m forex_ai_bot --status                 # Show bot status
    python3 -m forex_ai_bot --token YOUR_TOKEN --account YOUR_ACCOUNT_ID  # With OANDA credentials
    python3 -m forex_ai_bot --broker mt5             # Use MT5 instead (Windows)
"""

import argparse
import json
import sys
from .config import BotConfig, config
from .bot import ForexAIBot
from .report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="Forex AI Trading Bot - Ensemble ML with Continuous Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m forex_ai_bot --backtest                     Run backtest with simulated data
  python3 -m forex_ai_bot --backtest --symbol GBPUSD     Backtest on GBP/USD
  python3 -m forex_ai_bot --token xxx --account yyy      Start with OANDA credentials
  python3 -m forex_ai_bot --live --token xxx --account yyy   LIVE trading (real money!)
        """
    )

    # Mode
    parser.add_argument("--backtest", action="store_true", help="Run backtest on historical data")
    parser.add_argument("--train", action="store_true", help="Train/retrain models only")
    parser.add_argument("--status", action="store_true", help="Show bot and model status")
    parser.add_argument("--report", action="store_true", help="Show performance report + live trading readiness")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading (real money!)")

    # Backtest options
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol for backtest (default: EURUSD)")
    parser.add_argument("--bars", type=int, default=3000, help="Number of bars for backtest (default: 3000)")

    # Broker selection
    parser.add_argument("--broker", type=str, choices=["oanda", "mt5"], help="Broker to use (default: oanda)")

    # OANDA credentials
    parser.add_argument("--token", type=str, help="OANDA API token")
    parser.add_argument("--account", type=str, help="OANDA account ID (e.g., 101-001-12345678-001)")
    parser.add_argument("--practice", action="store_true", default=True, help="Use OANDA practice/demo account (default)")
    parser.add_argument("--no-practice", dest="practice", action="store_false", help="Use OANDA live account")

    # MT5 credentials (if using MT5)
    parser.add_argument("--login", type=int, help="MT5 account number")
    parser.add_argument("--password", type=str, help="MT5 password")
    parser.add_argument("--server", type=str, help="MT5 server")

    args = parser.parse_args()

    # Apply broker selection
    if args.broker:
        config.broker = args.broker

    # Apply OANDA credentials
    if args.token:
        config.oanda.api_token = args.token
    if args.account:
        config.oanda.account_id = args.account
    config.oanda.practice = args.practice

    # Apply MT5 credentials
    if args.login:
        config.mt5.login = args.login
    if args.password:
        config.mt5.password = args.password
    if args.server:
        config.mt5.server = args.server

    # Safety check for live trading
    if args.live:
        config.paper_trading = False
        config.mode = "live"
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
        print("!" * 60)
        print(f"  Broker:  {config.broker.upper()}")
        print(f"  Account: {config.oanda.account_id or config.mt5.login}")
        print()
        response = input("Type 'YES' to confirm live trading: ")
        if response != "YES":
            print("Cancelled. Use without --live for paper trading.")
            sys.exit(0)

    # Print startup banner
    print()
    print("=" * 50)
    print("  FOREX AI TRADING BOT")
    print(f"  Broker: {config.broker.upper()}")
    print(f"  Mode:   {'PAPER' if config.paper_trading else 'LIVE'}")
    print("=" * 50)
    print()

    bot = ForexAIBot(config)

    if args.report:
        generate_report()

    elif args.backtest:
        results = bot.backtest(args.symbol, args.bars)
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Symbol:        {results.get('symbol')}")
        print(f"Period:        {results.get('period')}")
        print(f"Total Trades:  {results.get('total_trades')}")
        print(f"Wins:          {results.get('wins')}")
        print(f"Losses:        {results.get('losses')}")
        print(f"Win Rate:      {results.get('win_rate', 0):.1f}%")
        print(f"Final Balance: ${results.get('final_balance', 10000):,.2f}")
        print(f"Return:        {results.get('return_pct', 0)}%")
        print("=" * 60)

    elif args.train:
        bot.connector.connect()
        metrics = bot.learner.train_models()
        bot.connector.disconnect()
        print("\nTraining Results:")
        print(json.dumps(metrics, indent=2, default=str))

    elif args.status:
        bot.connector.connect()
        status = bot.learner.get_learning_status()
        account = bot.connector.get_account_info()
        bot.connector.disconnect()
        print("\nBot Status:")
        print(json.dumps({**status, "account": account}, indent=2, default=str))

    else:
        bot.start()


if __name__ == "__main__":
    main()
