"""SQLite-backed trade journal.

Stores every signal we acted on plus its outcome. Two tables:

  trades         one row per opened position, updated on close
  signals        one row per emitted signal (acted-on or not, useful for analysis)

The journal is the *single source of truth* for the ML retrain pipeline and the
risk manager's win-rate stats. All columns are denormalised on purpose: it makes
ad-hoc analysis trivial in pandas / SQL.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .logging_setup import get_logger

log = get_logger(__name__)


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket          INTEGER UNIQUE,
        symbol          TEXT NOT NULL,
        side            TEXT NOT NULL,
        volume          REAL NOT NULL,
        entry_price     REAL NOT NULL,
        sl              REAL,
        tp              REAL,
        atr             REAL,
        risk_money      REAL,
        risk_pct        REAL,
        reason          TEXT,
        features_json   TEXT,
        opened_at       TEXT NOT NULL,    -- ISO UTC
        closed_at       TEXT,
        close_price     REAL,
        close_reason    TEXT,             -- "tp", "sl", "manual", "unknown"
        pnl             REAL,
        outcome         INTEGER,          -- 1 win, 0 loss, NULL while open
        magic           INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        bar_time        TEXT NOT NULL,
        symbol          TEXT NOT NULL,
        side            TEXT NOT NULL,
        reason          TEXT,
        entry_ref       REAL,
        atr             REAL,
        proba           REAL,             -- ML proba if available
        acted           INTEGER NOT NULL, -- 1 placed an order, 0 skipped
        skip_reason     TEXT,
        features_json   TEXT,
        created_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at)",
    "CREATE INDEX IF NOT EXISTS idx_trades_outcome   ON trades(outcome)",
    "CREATE INDEX IF NOT EXISTS idx_signals_bar_time ON signals(bar_time)",
]


@dataclass
class OpenTrade:
    ticket: int
    symbol: str
    side: str
    volume: float
    entry_price: float
    sl: float
    tp: float
    atr: float
    risk_money: float
    risk_pct: float
    reason: str
    features: Dict[str, float]
    opened_at: datetime
    magic: int


class Journal:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            for stmt in _SCHEMA:
                c.execute(stmt)

    # ------------------------------------------------------------------ inserts
    def record_signal(
        self,
        *,
        bar_time: datetime,
        symbol: str,
        side: str,
        reason: str,
        entry_ref: float,
        atr: float,
        proba: Optional[float],
        acted: bool,
        skip_reason: Optional[str],
        features: Optional[Dict[str, float]],
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO signals (
                    bar_time, symbol, side, reason, entry_ref, atr, proba,
                    acted, skip_reason, features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(bar_time), symbol, side, reason, float(entry_ref), float(atr),
                    float(proba) if proba is not None else None,
                    1 if acted else 0,
                    skip_reason,
                    json.dumps(features) if features else None,
                    _iso(datetime.now(tz=timezone.utc)),
                ),
            )
            return int(cur.lastrowid)

    def record_open(self, t: OpenTrade) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO trades (
                    ticket, symbol, side, volume, entry_price, sl, tp, atr,
                    risk_money, risk_pct, reason, features_json, opened_at, magic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t.ticket, t.symbol, t.side, t.volume, t.entry_price, t.sl, t.tp, t.atr,
                    t.risk_money, t.risk_pct, t.reason,
                    json.dumps(t.features) if t.features else None,
                    _iso(t.opened_at), t.magic,
                ),
            )

    def record_close(
        self,
        ticket: int,
        *,
        closed_at: datetime,
        close_price: float,
        close_reason: str,
        pnl: float,
    ) -> None:
        outcome = 1 if pnl > 0 else 0
        with self._conn() as c:
            c.execute(
                """
                UPDATE trades
                   SET closed_at = ?, close_price = ?, close_reason = ?,
                       pnl = ?, outcome = ?
                 WHERE ticket = ?
                """,
                (_iso(closed_at), float(close_price), close_reason,
                 float(pnl), outcome, int(ticket)),
            )

    def record_partial(
        self,
        ticket: int,
        *,
        closed_at: datetime,
        close_price: float,
        close_reason: str,
        pnl: float,
        volume: float,
    ) -> None:
        """Record a partial close as a *new* synthetic trade row so the parent
        position can stay open with reduced size. The synthetic row uses a
        negative ticket derived from the parent so:
          - PnL stats and equity curve include the realised partial PnL
          - The parent ticket remains visible as 'open' for the runner
          - Re-running record_partial on the same ticket produces unique rows
            (we suffix the row id by the partial sequence count).
        """
        # Determine the next partial sequence for this parent ticket.
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM trades "
                "  WHERE ticket < 0 AND symbol = (SELECT symbol FROM trades WHERE ticket = ?)"
                "    AND reason = ?",
                (int(ticket), f"partial_of_{ticket}"),
            ).fetchone()
            seq = int(row["n"] or 0) + 1
            partial_ticket = -(int(ticket) * 100 + seq)
            parent = c.execute(
                "SELECT symbol, side, entry_price, sl, tp, atr, risk_money, "
                "       risk_pct, opened_at, magic "
                "  FROM trades WHERE ticket = ?",
                (int(ticket),),
            ).fetchone()
            if parent is None:
                return
            outcome = 1 if pnl > 0 else 0
            c.execute(
                """
                INSERT INTO trades (
                    ticket, symbol, side, volume, entry_price, sl, tp, atr,
                    risk_money, risk_pct, reason, features_json,
                    opened_at, closed_at, close_price, close_reason,
                    pnl, outcome, magic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    partial_ticket,
                    parent["symbol"], parent["side"], float(volume),
                    parent["entry_price"], parent["sl"], parent["tp"],
                    parent["atr"], parent["risk_money"], parent["risk_pct"],
                    f"partial_of_{ticket}", None,
                    parent["opened_at"], _iso(closed_at),
                    float(close_price), close_reason,
                    float(pnl), outcome, parent["magic"],
                ),
            )

    # ------------------------------------------------------------------ queries
    def open_tickets(self, symbol: Optional[str] = None) -> List[int]:
        sql = "SELECT ticket FROM trades WHERE closed_at IS NULL"
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
            return [int(r["ticket"]) for r in rows]

    def last_loss_time(self, symbol: Optional[str] = None) -> Optional[datetime]:
        sql = "SELECT closed_at FROM trades WHERE outcome = 0"
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY closed_at DESC LIMIT 1"
        with self._conn() as c:
            row = c.execute(sql, params).fetchone()
            if not row or not row["closed_at"]:
                return None
            return _from_iso(row["closed_at"])

    def recent_outcomes(
        self,
        n: int = 30,
        side: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[int]:
        sql = "SELECT outcome FROM trades WHERE outcome IS NOT NULL"
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        if side:
            sql += " AND side = ?"
            params.append(side)
        sql += " ORDER BY closed_at DESC LIMIT ?"
        params.append(int(n))
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [int(r["outcome"]) for r in rows]

    def closed_count(self, symbol: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM trades WHERE outcome IS NOT NULL"
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        with self._conn() as c:
            row = c.execute(sql, params).fetchone()
            return int(row["n"])

    def closed_trades_df(self, symbol: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT * FROM trades WHERE outcome IS NOT NULL"
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY closed_at"
        with self._conn() as c:
            df = pd.read_sql_query(sql, c, params=params)
        return df

    # ------------------------------------------------------------------ export
    def export_csv(self, path: str | Path) -> None:
        df = self.closed_trades_df()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info("Exported %d closed trades to %s", len(df), path)


# ---------------------------------------------------------------------- helpers
def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)
