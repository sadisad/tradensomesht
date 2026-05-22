"""End-to-end test: spin up the dashboard with synthetic journal data,
hit every endpoint, and shut it down. Run with `python tests/dashboard_test.py`.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config import load_config
from bot.dashboard import create_app
from bot.indicators import FEATURE_COLUMNS
from bot.journal import Journal, OpenTrade


def seed_journal(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    j = Journal(db_path)
    base_time = datetime.now(tz=timezone.utc) - timedelta(days=2)
    for k in range(30):
        opened = base_time + timedelta(hours=k)
        closed = opened + timedelta(minutes=45)
        side = "buy" if k % 2 == 0 else "sell"
        outcome = 1 if k % 3 != 0 else 0
        ticket = 8_000_000 + k
        feats = {c: 0.1 * k for c in FEATURE_COLUMNS}
        feats["__side_buy"] = 1.0 if side == "buy" else 0.0
        j.record_open(OpenTrade(
            ticket=ticket, symbol="XAUUSD", side=side,
            volume=0.10, entry_price=2350.0 + (k - 15) * 0.5,
            sl=2345.0, tp=2360.0, atr=2.0,
            risk_money=50.0, risk_pct=0.5, reason="seed",
            features=feats, opened_at=opened, magic=42,
        ))
        pnl = 12.0 if outcome else -7.0
        j.record_close(
            ticket=ticket, closed_at=closed,
            close_price=(2360.0 if outcome else 2345.0),
            close_reason=("tp" if outcome else "sl"),
            pnl=pnl,
        )
        # Some skipped signals too
        if k % 4 == 0:
            j.record_signal(
                bar_time=opened, symbol="XAUUSD", side=side,
                reason="bull_cross" if side == "buy" else "bear_cross",
                entry_ref=2350.0, atr=2.0, proba=0.42,
                acted=False, skip_reason="ml_proba_below_threshold",
                features=feats,
            )
        j.record_signal(
            bar_time=opened, symbol="XAUUSD", side=side,
            reason="bull_cross" if side == "buy" else "bear_cross",
            entry_ref=2350.0, atr=2.0, proba=0.62,
            acted=True, skip_reason=None, features=feats,
        )
    # One open paper position
    open_t = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
    feats = {c: 0.0 for c in FEATURE_COLUMNS}
    j.record_open(OpenTrade(
        ticket=8_999_999, symbol="XAUUSD", side="buy",
        volume=0.05, entry_price=2350.0, sl=2347.0, tp=2356.0,
        atr=2.0, risk_money=15.0, risk_pct=0.5, reason="seed_open",
        features=feats, opened_at=open_t, magic=42,
    ))


def fetch(host: str, port: int, path: str) -> dict:
    url = f"http://{host}:{port}{path}"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "j.db"
        seed_journal(db)

        cfg = load_config()
        cfg["journal"]["db_path"] = str(db)
        # Force paper mode and fake-ish broker so MT5 isn't required
        cfg["trading"]["mode"] = "paper"
        cfg["broker"]["login"] = 0  # ensures we won't actually try logging in

        app = create_app(cfg)

        import uvicorn
        config = uvicorn.Config(app, host="127.0.0.1", port=8766, log_level="warning")
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()

        # Wait until the server is up
        for _ in range(40):
            try:
                fetch("127.0.0.1", 8766, "/api/stats")
                break
            except Exception:
                time.sleep(0.1)
        else:
            print("Dashboard failed to start")
            return 1

        endpoints = [
            "/api/status", "/api/stats", "/api/positions",
            "/api/trades?limit=10", "/api/signals?limit=10",
            "/api/equity_curve", "/api/chart_markers?limit=20",
        ]
        for ep in endpoints:
            data = fetch("127.0.0.1", 8766, ep)
            print(f"[OK] {ep:40s} keys={list(data.keys())[:6]}")

        # Spot checks
        stats = fetch("127.0.0.1", 8766, "/api/stats")
        assert stats["closed_trades"] == 30, stats
        positions = fetch("127.0.0.1", 8766, "/api/positions")
        assert len(positions["positions"]) == 1, positions
        trades = fetch("127.0.0.1", 8766, "/api/trades?limit=5")
        assert len(trades["trades"]) == 5, trades
        signals = fetch("127.0.0.1", 8766, "/api/signals?limit=5")
        assert len(signals["signals"]) == 5, signals
        markers = fetch("127.0.0.1", 8766, "/api/chart_markers?limit=10")
        assert len(markers["markers"]) > 0, markers

        # Verify the static index loads
        with urllib.request.urlopen("http://127.0.0.1:8766/", timeout=5) as r:
            html = r.read().decode("utf-8")
        assert "Robot Trading" in html, "index.html not served"
        print(f"[OK] / served {len(html)} bytes of HTML")

        server.should_exit = True
        t.join(timeout=3)
        print("\nAll dashboard tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
