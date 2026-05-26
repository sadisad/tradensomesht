"""Utility CLI: retrain the ML filter, export trades to CSV, print stats.

Examples:

  python -m bot.tools retrain
  python -m bot.tools export --out data/trades.csv
  python -m bot.tools stats
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .config import load_config
from .journal import Journal
from .logging_setup import get_logger, setup_logging
from .ml_filter import MLFilter

log = get_logger(__name__)


def cmd_retrain(cfg, symbol: Optional[str] = None) -> int:
    journal = Journal(cfg["journal"]["db_path"])
    ml = MLFilter(cfg["ml"])
    report = ml.retrain(journal, symbol=symbol)
    if report is None:
        print("Not enough data to retrain.")
        return 1
    scope = f"symbol={symbol}" if symbol else "all symbols"
    print(
        f"Retrained on {report.n_samples} samples ({scope}). "
        f"cv_auc={report.cv_auc:.3f} cv_acc={report.cv_acc:.3f} "
        f"balance={report.class_balance}"
    )
    return 0


def cmd_export(cfg, out: str) -> int:
    journal = Journal(cfg["journal"]["db_path"])
    journal.export_csv(out)
    print(f"Exported closed trades to {out}")
    return 0


def cmd_stats(cfg, symbol: Optional[str] = None) -> int:
    journal = Journal(cfg["journal"]["db_path"])
    df = journal.closed_trades_df(symbol=symbol)
    n = len(df)
    scope = f"symbol={symbol}" if symbol else "all symbols"
    if n == 0:
        print(f"No closed trades yet ({scope}).")
        return 0
    wins = int((df["outcome"] == 1).sum())
    losses = int((df["outcome"] == 0).sum())
    pnl = float(df["pnl"].sum()) if "pnl" in df else 0.0
    avg_pnl = float(df["pnl"].mean()) if "pnl" in df else 0.0
    print(f"[{scope}] closed={n} wins={wins} losses={losses} win_rate={wins / n:.1%}")
    print(f"  total_pnl={pnl:.2f} avg_pnl_per_trade={avg_pnl:.2f}")
    if "side" in df:
        for side, sub in df.groupby("side"):
            sw = int((sub["outcome"] == 1).sum())
            print(f"  {side}: n={len(sub)} wins={sw} wr={sw / len(sub):.1%} pnl={sub['pnl'].sum():.2f}")
    # If no symbol filter, also break down per-symbol
    if not symbol and "symbol" in df:
        print("\nPer symbol:")
        for sym, sub in df.groupby("symbol"):
            sw = int((sub["outcome"] == 1).sum())
            print(
                f"  {sym}: n={len(sub)} wins={sw} wr={sw / len(sub):.1%} "
                f"pnl={sub['pnl'].sum():.2f}"
            )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Axiom Omega utilities")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--symbol",
        default=None,
        help="Restrict retrain/stats to one symbol (default: all symbols)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("retrain", help="Force-retrain the ML filter from the journal")
    p_export = sub.add_parser("export", help="Export closed trades to CSV")
    p_export.add_argument("--out", required=True, help="Output CSV path")
    sub.add_parser("stats", help="Print summary stats from the journal")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.cmd == "retrain":
        return cmd_retrain(cfg, symbol=args.symbol)
    if args.cmd == "export":
        return cmd_export(cfg, args.out)
    if args.cmd == "stats":
        return cmd_stats(cfg, symbol=args.symbol)
    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
