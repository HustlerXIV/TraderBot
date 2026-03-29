"""
Performance Report Generator
Run anytime to see how the bot is performing:
    python3 -m forex_ai_bot --report
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from .config import MODEL_DIR, LOG_DIR


def generate_report():
    """Print a full performance report to the terminal."""

    print()
    print("=" * 60)
    print("  FOREX AI BOT - PERFORMANCE REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Trade History ──────────────────────────────────────
    trade_log = _load_trade_log()
    _print_trade_summary(trade_log)

    # ── 2. Model Performance ──────────────────────────────────
    perf_log = _load_performance_log()
    _print_model_performance(perf_log)

    # ── 3. Readiness for Live Trading ────────────────────────
    _print_live_trading_readiness(trade_log, perf_log)

    print("=" * 60)
    print()


def _load_trade_log() -> List[Dict]:
    """Load trade history from log file."""
    trades = []
    log_file = LOG_DIR / "trading_bot.log"

    if not log_file.exists():
        return trades

    with open(log_file, "r") as f:
        for line in f:
            if "PAPER TRADE" in line or "Trade closed" in line or "Order filled" in line:
                trades.append({"raw": line.strip(), "time": line[:19]})

    return trades


def _load_performance_log() -> List[Dict]:
    """Load ML performance log."""
    perf_file = MODEL_DIR / "performance_log.json"
    if not perf_file.exists():
        return []

    try:
        with open(perf_file, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _print_trade_summary(trades: List[Dict]):
    """Print trade statistics."""
    print()
    print("  TRADING ACTIVITY")
    print("  " + "-" * 40)

    log_file = LOG_DIR / "trading_bot.log"
    if not log_file.exists():
        print("  No trading log found yet.")
        print("  Start the bot first: python3 -m forex_ai_bot")
        return

    # Count activity from log
    buys = sells = closed = retrained = errors = 0
    with open(log_file, "r") as f:
        for line in f:
            if "BUY" in line and "PAPER" in line:
                buys += 1
            elif "SELL" in line and "PAPER" in line:
                sells += 1
            elif "Trade closed" in line or "closed" in line.lower():
                closed += 1
            elif "training complete" in line.lower() or "retrain" in line.lower():
                retrained += 1
            elif "ERROR" in line:
                errors += 1

    # Get log file age
    log_age_days = (datetime.now().timestamp() - log_file.stat().st_mtime) / 86400
    log_size_kb = log_file.stat().st_size / 1024

    print(f"  Log file size:   {log_size_kb:.1f} KB")
    print(f"  BUY signals:     {buys}")
    print(f"  SELL signals:    {sells}")
    print(f"  Closed trades:   {closed}")
    print(f"  Model retrains:  {retrained}")
    print(f"  Errors logged:   {errors}")


def _print_model_performance(perf_log: List[Dict]):
    """Print ML model performance over time."""
    print()
    print("  MODEL PERFORMANCE OVER TIME")
    print("  " + "-" * 40)

    if not perf_log:
        print("  No performance data yet.")
        print("  The bot records this after live paper trading.")
        return

    # Show last 5 entries
    recent = perf_log[-5:]
    for entry in recent:
        ts = entry.get("timestamp", "")[:16]
        trades = entry.get("total_trades", 0)
        win_rate = entry.get("win_rate", 0) * 100
        pnl = entry.get("total_pnl", 0)
        print(f"  {ts}  |  Trades: {trades:3d}  |  Win Rate: {win_rate:.1f}%  |  PnL: {pnl:+.2f}")

    # Trend
    if len(perf_log) >= 3:
        early_wr = perf_log[0].get("win_rate", 0) * 100
        latest_wr = perf_log[-1].get("win_rate", 0) * 100
        trend = latest_wr - early_wr
        arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
        print(f"\n  Win Rate Trend:  {arrow} {trend:+.1f}% since start")


def _print_live_trading_readiness(trades: List[Dict], perf_log: List[Dict]):
    """Print a checklist for live trading readiness."""
    print()
    print("  LIVE TRADING READINESS CHECKLIST")
    print("  " + "-" * 40)

    checks = []

    # Check 1: Has the bot been running?
    log_file = LOG_DIR / "trading_bot.log"
    has_log = log_file.exists() and log_file.stat().st_size > 1000
    checks.append(("Bot has been running", has_log))

    # Check 2: Enough performance data (at least 2 weeks worth)
    has_enough_data = len(perf_log) >= 14
    checks.append(("2+ weeks of performance data", has_enough_data))

    # Check 3: Recent win rate above 40%
    recent_wr = 0
    if perf_log:
        recent_entries = perf_log[-7:]  # Last 7 entries
        recent_wr = sum(e.get("win_rate", 0) for e in recent_entries) / len(recent_entries) * 100
    has_good_wr = recent_wr >= 40
    checks.append((f"Win rate ≥ 40% (current: {recent_wr:.1f}%)", has_good_wr))

    # Check 4: Positive total PnL
    total_pnl = sum(e.get("total_pnl", 0) for e in perf_log)
    has_positive_pnl = total_pnl > 0
    checks.append((f"Positive total PnL (current: {total_pnl:+.2f})", has_positive_pnl))

    # Check 5: No recent halt
    model_dir_exists = MODEL_DIR.exists()
    checks.append(("Models saved successfully", model_dir_exists))

    # Print checklist
    all_passed = True
    for label, passed in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {label}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🟢 READY FOR LIVE TRADING")
        print("  Run: python3 -m forex_ai_bot --live --token X --account Y")
    else:
        remaining = sum(1 for _, p in checks if not p)
        print(f"  🔴 NOT READY YET — {remaining} check(s) still failing")
        print("  Keep running paper trading and check back later.")
