"""Quick journal + log inspection for multi-pair debug."""
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "journal.db"
DATA = ROOT / "data"


def main() -> None:
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    print("=== signals by symbol ===")
    for r in c.execute("SELECT symbol, COUNT(*) AS n FROM signals GROUP BY symbol"):
        print(f"  {r['symbol']:10s} {r['n']:4d}")
    if c.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 0:
        print("  (empty)")

    print()
    print("=== trades by symbol ===")
    for r in c.execute("SELECT symbol, COUNT(*) AS n FROM trades GROUP BY symbol"):
        print(f"  {r['symbol']:10s} {r['n']:4d}")
    if c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0:
        print("  (empty)")

    print()
    print("=== last 5 signals (any symbol) ===")
    for r in c.execute(
        "SELECT bar_time, symbol, side, reason, acted, skip_reason "
        "FROM signals ORDER BY id DESC LIMIT 5"
    ):
        print(f"  {r['bar_time']} {r['symbol']:8s} {r['side']:4s} {r['reason']:18s} "
              f"acted={r['acted']} skip={r['skip_reason']}")

    print()
    print("=== bot logs ===")
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".log"):
            path = DATA / f
            size = path.stat().st_size
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            print(f"  {f:30s} size={size:8d} bytes  modified={mtime.isoformat()}")


if __name__ == "__main__":
    main()
