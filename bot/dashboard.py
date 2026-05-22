"""FastAPI dashboard for the trading bot.

Serves a single HTML page (``static/index.html``) plus JSON endpoints that read
from the SQLite journal and (optionally) the MT5 terminal.

Designed to run as a *separate process* alongside the live bot. Both share the
same journal file, so the dashboard always reflects the bot's actual state.

Run:

    python -m bot.dashboard

By default it binds 127.0.0.1:8765. Override with ``--host`` / ``--port``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config, project_root
from .logging_setup import get_logger, setup_logging

log = get_logger(__name__)

# Lazy import: only needed when MT5 is reachable. The dashboard still works
# without the terminal (it just hides live price + equity).
try:  # pragma: no cover
    from .broker_mt5 import MT5Client
except Exception:  # noqa: BLE001
    MT5Client = None  # type: ignore


# ---------------------------------------------------------------------- state
class State:
    """Process-wide singletons. Populated in ``create_app``."""

    cfg: Dict[str, Any] = {}
    db_path: Path = Path("data/journal.db")
    mt5_client: Optional[Any] = None
    mt5_connected: bool = False


STATE = State()


# ---------------------------------------------------------------------- helpers
@contextmanager
def _db():
    conn = sqlite3.connect(STATE.db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _try_mt5():
    """Best-effort connect / reconnect. Failures are non-fatal."""
    if MT5Client is None:
        return None
    if STATE.mt5_client is None:
        try:
            STATE.mt5_client = MT5Client(STATE.cfg.get("broker", {}))
            STATE.mt5_client.connect()
            STATE.mt5_connected = True
            log.info("Dashboard connected to MT5")
        except Exception as e:  # noqa: BLE001
            log.warning("MT5 unavailable for dashboard: %s", e)
            STATE.mt5_client = None
            STATE.mt5_connected = False
    return STATE.mt5_client


def _row_to_dict(r: sqlite3.Row) -> Dict[str, Any]:
    return {k: r[k] for k in r.keys()}


def _available_symbols(default: str) -> List[str]:
    """Symbols this dashboard knows about: union of journal-observed symbols + config default."""
    syms = {default} if default else set()
    try:
        with _db() as c:
            for r in c.execute("SELECT DISTINCT symbol FROM trades WHERE symbol IS NOT NULL"):
                syms.add(str(r["symbol"]))
            for r in c.execute("SELECT DISTINCT symbol FROM signals WHERE symbol IS NOT NULL"):
                syms.add(str(r["symbol"]))
    except Exception as e:  # noqa: BLE001
        log.debug("symbol discovery failed (journal not ready?): %s", e)
    return sorted(s for s in syms if s)


def _resolve_symbol(requested: Optional[str], default: str) -> str:
    """Sanitize a user-supplied symbol. We accept anything alphanumeric + a few separators
    so brokers' suffixed names (XAUUSD.s, EURUSD-pro) work; reject anything else to avoid
    feeding garbage to MT5 or SQL."""
    if not requested:
        return default
    cleaned = requested.strip()
    if not cleaned or len(cleaned) > 32:
        return default
    if not all(ch.isalnum() or ch in "._-" for ch in cleaned):
        return default
    return cleaned


# ---------------------------------------------------------------------- app factory
def create_app(cfg: Dict[str, Any]) -> FastAPI:
    STATE.cfg = cfg
    STATE.db_path = Path(cfg.get("journal", {}).get("db_path", "data/journal.db"))

    app = FastAPI(title="Robot Trading Dashboard", version="0.1.0")
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --------------------------------------------------------------- routes
    @app.get("/", include_in_schema=False)
    def index():
        idx = static_dir / "index.html"
        if not idx.exists():
            raise HTTPException(500, "index.html not found")
        return FileResponse(idx)

    @app.get("/api/symbols")
    def api_symbols():
        """List of symbols the dashboard can switch between (journal + config default)."""
        default = cfg["trading"]["symbol"]
        return {"default": default, "symbols": _available_symbols(default)}

    @app.get("/api/status")
    def api_status(symbol: Optional[str] = None):
        client = _try_mt5()
        default_symbol = cfg["trading"]["symbol"]
        sym = _resolve_symbol(symbol, default_symbol)
        timeframe = cfg["trading"]["timeframe"]
        mode = cfg["trading"].get("mode", "paper")
        out: Dict[str, Any] = {
            "symbol": sym,
            "timeframe": timeframe,
            "mode": mode,
            "magic": cfg["broker"].get("magic"),
            "mt5_connected": False,
            "balance": None,
            "equity": None,
            "currency": None,
            "server": None,
            "last_price": None,
            "now_utc": datetime.now(tz=timezone.utc).isoformat(),
        }
        if client is None:
            return out
        try:
            import MetaTrader5 as mt5  # type: ignore
            info = mt5.account_info()
            tick = mt5.symbol_info_tick(sym)
            out["mt5_connected"] = True
            if info is not None:
                out["balance"] = float(info.balance)
                out["equity"] = float(info.equity)
                out["currency"] = str(info.currency)
                out["server"] = str(info.server)
            if tick is not None:
                out["last_price"] = float((tick.ask + tick.bid) / 2.0)
                out["bid"] = float(tick.bid)
                out["ask"] = float(tick.ask)
        except Exception as e:  # noqa: BLE001
            log.warning("status: mt5 query failed: %s", e)
        return out

    @app.get("/api/candles")
    def api_candles(
        bars: int = Query(500, ge=1, le=5000),
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        client = _try_mt5()
        if client is None:
            return JSONResponse({"error": "mt5_unavailable", "candles": []}, status_code=200)
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        tf = timeframe or cfg["trading"]["timeframe"]
        try:
            df = client.get_rates(sym, tf, bars)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e), "candles": []}, status_code=200)
        candles = [
            {
                "time": int(idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for idx, row in df.iterrows()
        ]
        return {"symbol": sym, "timeframe": tf, "candles": candles}

    @app.get("/api/positions")
    def api_positions(symbol: Optional[str] = None):
        """Open positions: from MT5 in demo mode, from journal in paper mode."""
        mode = cfg["trading"].get("mode", "paper")
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        if mode == "demo":
            client = _try_mt5()
            if client is None:
                return {"mode": mode, "positions": []}
            try:
                pos = client.open_positions(sym)
            except Exception as e:  # noqa: BLE001
                return {"mode": mode, "positions": [], "error": str(e)}
            return {
                "mode": mode,
                "positions": [
                    {
                        "ticket": p.ticket,
                        "side": p.side,
                        "volume": p.volume,
                        "entry": p.price_open,
                        "sl": p.sl,
                        "tp": p.tp,
                        "pnl": p.profit,
                        "opened_at": p.time_open.isoformat(),
                    }
                    for p in pos
                ],
            }
        # paper mode: read open rows from the journal, filtered by symbol
        with _db() as c:
            rows = c.execute(
                "SELECT ticket, side, volume, entry_price AS entry, sl, tp, "
                "       opened_at, atr "
                "  FROM trades WHERE closed_at IS NULL AND symbol = ? "
                " ORDER BY opened_at DESC",
                (sym,),
            ).fetchall()
        return {"mode": mode, "positions": [_row_to_dict(r) for r in rows]}

    @app.get("/api/trades")
    def api_trades(
        limit: int = Query(50, ge=1, le=500),
        symbol: Optional[str] = None,
    ):
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        with _db() as c:
            rows = c.execute(
                "SELECT ticket, symbol, side, volume, entry_price, sl, tp, atr, "
                "       risk_money, risk_pct, reason, opened_at, closed_at, "
                "       close_price, close_reason, pnl, outcome "
                "  FROM trades WHERE closed_at IS NOT NULL AND symbol = ? "
                " ORDER BY closed_at DESC LIMIT ?",
                (sym, int(limit)),
            ).fetchall()
        return {"trades": [_row_to_dict(r) for r in rows]}

    @app.get("/api/signals")
    def api_signals(
        limit: int = Query(50, ge=1, le=500),
        symbol: Optional[str] = None,
    ):
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        with _db() as c:
            rows = c.execute(
                "SELECT id, bar_time, symbol, side, reason, entry_ref, atr, "
                "       proba, acted, skip_reason, created_at "
                "  FROM signals WHERE symbol = ? "
                " ORDER BY id DESC LIMIT ?",
                (sym, int(limit)),
            ).fetchall()
        return {"signals": [_row_to_dict(r) for r in rows]}

    @app.get("/api/equity_curve")
    def api_equity_curve(symbol: Optional[str] = None):
        """Cumulative pnl over closed trades (UTC). If symbol is given, scope to that pair;
        otherwise show portfolio-wide cumulative pnl across all symbols."""
        sym = _resolve_symbol(symbol, "") or None
        with _db() as c:
            if sym:
                rows = c.execute(
                    "SELECT closed_at, pnl FROM trades "
                    " WHERE closed_at IS NOT NULL AND symbol = ? "
                    " ORDER BY closed_at",
                    (sym,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT closed_at, pnl FROM trades "
                    " WHERE closed_at IS NOT NULL "
                    " ORDER BY closed_at"
                ).fetchall()
        cum = 0.0
        points = []
        for r in rows:
            cum += float(r["pnl"] or 0.0)
            points.append({"time": r["closed_at"], "cum_pnl": cum})
        return {"symbol": sym, "points": points}

    @app.get("/api/stats")
    def api_stats(symbol: Optional[str] = None):
        """Per-symbol stats when ?symbol= is given, otherwise portfolio-wide."""
        sym = _resolve_symbol(symbol, "") or None
        where_sym = " AND symbol = ?" if sym else ""
        params = (sym,) if sym else ()
        with _db() as c:
            row = c.execute(
                "SELECT COUNT(*)               AS n,"
                "       SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) AS wins,"
                "       SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END) AS losses,"
                "       COALESCE(SUM(pnl), 0)  AS total_pnl,"
                "       COALESCE(AVG(pnl), 0)  AS avg_pnl"
                "  FROM trades WHERE outcome IS NOT NULL" + where_sym,
                params,
            ).fetchone()
            today_iso = (datetime.now(tz=timezone.utc) - timedelta(hours=24)).isoformat()
            today = c.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(pnl),0) AS pnl "
                "  FROM trades WHERE closed_at IS NOT NULL AND closed_at > ?" + where_sym,
                (today_iso, *params),
            ).fetchone()
            sig = c.execute(
                "SELECT COUNT(*) AS n,"
                "       SUM(CASE WHEN acted=1 THEN 1 ELSE 0 END) AS acted "
                "  FROM signals" + (" WHERE symbol = ?" if sym else ""),
                params,
            ).fetchone()
            last_loss = c.execute(
                "SELECT closed_at FROM trades WHERE outcome=0" + where_sym +
                " ORDER BY closed_at DESC LIMIT 1",
                params,
            ).fetchone()
        n = int(row["n"] or 0)
        wins = int(row["wins"] or 0)
        return {
            "symbol": sym,
            "closed_trades": n,
            "wins": wins,
            "losses": int(row["losses"] or 0),
            "win_rate": (wins / n) if n else 0.0,
            "total_pnl": float(row["total_pnl"] or 0.0),
            "avg_pnl": float(row["avg_pnl"] or 0.0),
            "trades_24h": int(today["n"] or 0),
            "pnl_24h": float(today["pnl"] or 0.0),
            "signals_total": int(sig["n"] or 0),
            "signals_acted": int(sig["acted"] or 0),
            "last_loss_at": last_loss["closed_at"] if last_loss else None,
        }

    @app.get("/api/chart_markers")
    def api_chart_markers(
        limit: int = Query(100, ge=1, le=500),
        symbol: Optional[str] = None,
    ):
        """Trade entries/exits as chart markers (most recent first), filtered by symbol."""
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        with _db() as c:
            rows = c.execute(
                "SELECT ticket, side, entry_price, opened_at, "
                "       close_price, closed_at, close_reason, outcome, reason "
                "  FROM trades WHERE symbol = ? ORDER BY opened_at DESC LIMIT ?",
                (sym, int(limit)),
            ).fetchall()
        markers: List[Dict[str, Any]] = []
        for r in rows:
            try:
                t_in = int(datetime.fromisoformat(r["opened_at"]).timestamp())
            except Exception:  # noqa: BLE001
                continue
            markers.append({
                "time": t_in,
                "position": "belowBar" if r["side"] == "buy" else "aboveBar",
                "color": "#0d8a6e" if r["side"] == "buy" else "#b8423a",
                "shape": "arrowUp" if r["side"] == "buy" else "arrowDown",
                "text": f"{r['side']} #{r['ticket']}",
            })
            if r["closed_at"]:
                try:
                    t_out = int(datetime.fromisoformat(r["closed_at"]).timestamp())
                except Exception:  # noqa: BLE001
                    continue
                outcome_color = "#0d8a6e" if r["outcome"] == 1 else "#b8423a"
                markers.append({
                    "time": t_out,
                    "position": "aboveBar" if r["side"] == "buy" else "belowBar",
                    "color": outcome_color,
                    "shape": "circle",
                    "text": f"close {r['close_reason'] or ''}".strip(),
                })
        # Sort ascending by time for chart consumption
        markers.sort(key=lambda m: m["time"])
        return {"markers": markers}

    return app


# ---------------------------------------------------------------------- entrypoint
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Robot Trading dashboard")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    app = create_app(cfg)

    import uvicorn
    log.info("Dashboard starting on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
