"""CLI to backtest the strategy on historical CSV data or live MT5 history.

Examples:

  # From a CSV file with columns: time,open,high,low,close,volume
  python -m bot.run_backtest --csv data/xauusd_m5.csv

  # Pulled from a connected MT5 terminal
  python -m bot.run_backtest --from-mt5 --bars 20000

The CSV ``time`` column is parsed as UTC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .backtest import run_backtest
from .config import load_config
from .logging_setup import get_logger, setup_logging

log = get_logger(__name__)


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.rename(columns={cols[c]: c for c in cols})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _load_from_mt5(cfg, n_bars: int) -> pd.DataFrame:
    from .broker_mt5 import MT5Client
    client = MT5Client(cfg["broker"])
    client.connect()
    try:
        return client.get_rates(cfg["trading"]["symbol"], cfg["trading"]["timeframe"], n_bars)
    finally:
        client.disconnect()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the EMA+RSI+ATR strategy")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--csv", default=None, help="OHLCV CSV file path")
    parser.add_argument("--from-mt5", action="store_true", help="Pull bars from MT5 terminal")
    parser.add_argument("--bars", type=int, default=20000, help="Bars to pull from MT5")
    parser.add_argument("--spread", type=float, default=0.0, help="Spread in price units (per side)")
    parser.add_argument("--out", default=None, help="Optional CSV path to dump trades")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.csv:
        df = _load_csv(args.csv)
    elif args.from_mt5:
        df = _load_from_mt5(cfg, args.bars)
    else:
        parser.error("Provide --csv PATH or --from-mt5")
        return 2

    log.info("Backtesting on %d bars (%s -> %s)", len(df), df.index[0], df.index[-1])
    report = run_backtest(df, cfg["strategy"], cfg["risk"], spread_price=args.spread)
    print(report.summary())

    if args.out:
        rows = []
        for t in report.trades:
            rows.append({
                "bar_in": t.bar_in, "bar_out": t.bar_out,
                "side": t.side, "entry": t.entry, "sl": t.sl, "tp": t.tp,
                "exit": t.exit, "bars_held": t.bars_held,
                "pnl_price": t.pnl_price, "outcome": t.outcome, "reason": t.reason,
            })
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        log.info("Wrote %d trades to %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
