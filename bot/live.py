"""Main live trading loop.

Responsibilities per cycle:

  1. Sync open positions: detect any positions closed since last loop (SL/TP hits)
     and update the journal with the realised PnL.
  2. Pull bars + compute indicators + features.
  3. Ask the strategy for a signal on the last closed bar.
  4. If a signal exists:
       a. Build a feature snapshot (for ML + journal).
       b. Score with the ML filter.
       c. Apply pre-trade gates (hours, cooldown, daily loss, max positions).
       d. Build a TradePlan via the risk manager (with dynamic risk scaling).
       e. In paper mode: log + journal as a synthetic open. In demo mode: send order.
  5. Periodically retrain the ML filter from the journal.

The loop is intentionally single-threaded and stateless between iterations
(state lives in MT5 + the SQLite journal). That makes restarts safe.
"""
from __future__ import annotations

import argparse
import math
import signal as os_signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .broker_mt5 import MT5Client, Position
from .config import load_config
from .indicators import build_features
from .journal import Journal, OpenTrade
from .logging_setup import get_logger, setup_logging
from .ml_filter import MLFilter
from .risk import RiskManager, RiskRejected
from .strategy import EmaRsiAtrStrategy

log = get_logger(__name__)


class LiveBot:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.broker_cfg = cfg["broker"]
        self.trading_cfg = cfg["trading"]
        self.risk_cfg = cfg["risk"]
        self.strategy_cfg = cfg["strategy"]
        self.ml_cfg = cfg["ml"]
        self.journal_cfg = cfg["journal"]

        self.symbol: str = self.trading_cfg["symbol"]
        self.timeframe: str = self.trading_cfg["timeframe"]
        self.history_bars: int = int(self.trading_cfg["history_bars"])
        self.loop_seconds: int = int(self.trading_cfg["loop_seconds"])
        self.mode: str = str(self.trading_cfg.get("mode", "paper")).lower()
        if self.mode not in ("paper", "demo"):
            raise ValueError(f"Unsupported trading.mode: {self.mode}")

        self.client = MT5Client(self.broker_cfg)
        self.strategy = EmaRsiAtrStrategy(self.strategy_cfg, self.risk_cfg)
        self.risk = RiskManager(self.risk_cfg, self.trading_cfg)
        self.journal = Journal(self.journal_cfg["db_path"])
        self.ml = MLFilter(self.ml_cfg)

        self._last_bar_time: Optional[pd.Timestamp] = None
        self._last_deals_check: datetime = datetime.now(tz=timezone.utc) - timedelta(days=1)
        self._start_of_day_equity: float = 0.0
        self._start_of_day_date: Optional[datetime] = None
        self._stop = False

        # Paper-mode in-memory positions (we don't really place orders)
        self._paper_positions: Dict[int, Dict[str, Any]] = {}
        self._paper_next_ticket: int = 9_000_000_000

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        log.info(
            "Starting LiveBot symbol=%s tf=%s mode=%s magic=%s",
            self.symbol, self.timeframe, self.mode, self.broker_cfg.get("magic"),
        )
        self.client.connect()
        try:
            self._refresh_day_anchor(force=True)
            while not self._stop:
                try:
                    self.cycle()
                except Exception:  # noqa: BLE001
                    log.exception("cycle failed")
                self._sleep(self.loop_seconds)
        finally:
            self.client.disconnect()
            log.info("LiveBot stopped")

    def stop(self) -> None:
        self._stop = True

    def _sleep(self, seconds: int) -> None:
        end = time.time() + seconds
        while not self._stop and time.time() < end:
            time.sleep(0.5)

    # ------------------------------------------------------------------ main cycle
    def cycle(self) -> None:
        self._refresh_day_anchor()
        # 1. Reconcile any closed positions from the broker side
        self._reconcile_closed()

        # In paper mode also resolve synthetic positions against the latest bar
        if self.mode == "paper":
            self._resolve_paper_positions()

        # 2. Pull bars
        df = self.client.get_rates(self.symbol, self.timeframe, self.history_bars)
        if len(df) < int(self.strategy_cfg["ema_trend"]) + 5:
            log.warning("Not enough bars yet (%d)", len(df))
            return
        ind = self.strategy.prepare(df)

        # Avoid reacting to the same bar twice
        last_closed_bar_time = ind.index[-2]
        if self._last_bar_time is not None and last_closed_bar_time <= self._last_bar_time:
            return
        self._last_bar_time = last_closed_bar_time

        # 3. Strategy signal
        sig = self.strategy.evaluate(ind)
        if sig is None:
            return

        # 4. Build features for ML + journal
        feats_df = build_features(ind)
        feats_row = feats_df.iloc[-2]
        features: Dict[str, float] = {
            k: (float(v) if pd.notna(v) else float("nan")) for k, v in feats_row.items()
        }
        features["__side_buy"] = 1.0 if sig.side == "buy" else 0.0

        proba = self.ml.predict_proba_win(features)
        log.info(
            "Signal %s %s @ %.5f atr=%.5f reason=%s ml_proba=%.3f bar=%s",
            sig.side, self.symbol, sig.entry, sig.atr, sig.reason, proba, sig.bar_time,
        )

        # 5. Pre-trade gates
        skip_reason = self._pre_trade_block_reason(proba)
        if skip_reason is not None:
            self.journal.record_signal(
                bar_time=sig.bar_time.to_pydatetime(),
                symbol=self.symbol, side=sig.side, reason=sig.reason,
                entry_ref=sig.entry, atr=sig.atr, proba=proba,
                acted=False, skip_reason=skip_reason, features=features,
            )
            log.info("Signal skipped: %s", skip_reason)
            return

        # 6. Build trade plan (with dynamic risk)
        try:
            sym_info = self.client.symbol_info(self.symbol)
            equity = self.client.account_equity()
            recent = self.journal.recent_outcomes(n=30, side=sig.side)
            risk_scale = self.risk.risk_scale_from_history(recent)
            plan = self.risk.build_plan(
                side=sig.side,
                entry=sig.entry,
                atr_value=sig.atr,
                equity=equity,
                symbol_info=sym_info,
                risk_scale=risk_scale,
            )
        except RiskRejected as e:
            self.journal.record_signal(
                bar_time=sig.bar_time.to_pydatetime(),
                symbol=self.symbol, side=sig.side, reason=sig.reason,
                entry_ref=sig.entry, atr=sig.atr, proba=proba,
                acted=False, skip_reason=f"risk_rejected:{e}", features=features,
            )
            log.warning("Risk manager rejected trade: %s", e)
            return

        log.info(
            "Plan side=%s vol=%.2f entry=%.5f sl=%.5f tp=%.5f risk=%.2f$ (%.2f%%) scale=%.2f",
            plan.side, plan.volume, plan.entry, plan.sl, plan.tp,
            plan.risk_money, plan.risk_pct, risk_scale,
        )

        # 7. Execute
        self.journal.record_signal(
            bar_time=sig.bar_time.to_pydatetime(),
            symbol=self.symbol, side=sig.side, reason=sig.reason,
            entry_ref=sig.entry, atr=sig.atr, proba=proba,
            acted=True, skip_reason=None, features=features,
        )

        if self.mode == "paper":
            self._place_paper(sig, plan, features, proba)
        else:
            self._place_demo(sig, plan, features, proba)

        # 8. Maybe retrain
        report = self.ml.maybe_retrain(self.journal)
        if report is not None:
            log.info("ML retrain summary: %s", report)

    # ------------------------------------------------------------------ gates
    def _pre_trade_block_reason(self, proba: float) -> Optional[str]:
        if not self.risk.within_trading_hours():
            return "outside_trading_hours"
        last_loss = self.journal.last_loss_time()
        if self.risk.cooldown_active(last_loss):
            return "cooldown_after_loss"
        equity = self.client.account_equity() if self.mode == "demo" else self._start_of_day_equity
        if self.risk.daily_loss_breached(self._start_of_day_equity, equity):
            return "daily_loss_limit"
        # Existing positions cap
        max_pos = int(self.trading_cfg.get("max_open_positions", 1))
        n_open = self._count_open_positions()
        if n_open >= max_pos:
            return f"max_open_positions({n_open}>={max_pos})"
        # ML threshold
        if not self.ml.should_trade(proba):
            return f"ml_proba_below_threshold({proba:.3f}<{self.ml.min_proba:.3f})"
        return None

    def _count_open_positions(self) -> int:
        if self.mode == "paper":
            return len(self._paper_positions)
        return len(self.client.open_positions(self.symbol))

    def _refresh_day_anchor(self, force: bool = False) -> None:
        now = datetime.now(tz=timezone.utc)
        today = now.date()
        if force or self._start_of_day_date != today:
            try:
                self._start_of_day_equity = self.client.account_equity()
            except Exception:  # noqa: BLE001
                self._start_of_day_equity = 0.0
            self._start_of_day_date = today
            log.info(
                "Day anchor set: %s equity=%.2f",
                today, self._start_of_day_equity,
            )

    # ------------------------------------------------------------------ execution
    def _place_demo(self, sig, plan, features: Dict[str, float], proba: float) -> None:
        result = self.client.place_market_order(
            symbol=self.symbol,
            side=plan.side,
            volume=plan.volume,
            sl=plan.sl,
            tp=plan.tp,
            comment=f"r:{sig.reason[:10]}",
        )
        if not result.ok or result.ticket is None:
            log.error("Order failed; not journaling open. retcode=%s", result.retcode)
            return
        self.journal.record_open(
            OpenTrade(
                ticket=int(result.ticket),
                symbol=self.symbol,
                side=plan.side,
                volume=plan.volume,
                entry_price=float(result.price or plan.entry),
                sl=plan.sl, tp=plan.tp,
                atr=sig.atr,
                risk_money=plan.risk_money,
                risk_pct=plan.risk_pct,
                reason=sig.reason,
                features=features,
                opened_at=datetime.now(tz=timezone.utc),
                magic=int(self.broker_cfg.get("magic", 0)),
            )
        )

    def _place_paper(self, sig, plan, features: Dict[str, float], proba: float) -> None:
        self._paper_next_ticket += 1
        ticket = self._paper_next_ticket
        opened_at = datetime.now(tz=timezone.utc)
        self._paper_positions[ticket] = {
            "side": plan.side,
            "entry": plan.entry,
            "sl": plan.sl,
            "tp": plan.tp,
            "volume": plan.volume,
            "opened_at": opened_at,
            "atr": sig.atr,
        }
        self.journal.record_open(
            OpenTrade(
                ticket=ticket,
                symbol=self.symbol,
                side=plan.side,
                volume=plan.volume,
                entry_price=plan.entry,
                sl=plan.sl, tp=plan.tp,
                atr=sig.atr,
                risk_money=plan.risk_money,
                risk_pct=plan.risk_pct,
                reason=sig.reason,
                features=features,
                opened_at=opened_at,
                magic=int(self.broker_cfg.get("magic", 0)),
            )
        )
        log.info("[paper] opened ticket=%s %s vol=%.2f", ticket, plan.side, plan.volume)

    # ------------------------------------------------------------------ reconciliation
    def _reconcile_closed(self) -> None:
        """Detect demo positions that closed since last check and update the journal."""
        if self.mode != "demo":
            return
        deals = self.client.deals_since(self._last_deals_check - timedelta(minutes=1))
        if not deals:
            self._last_deals_check = datetime.now(tz=timezone.utc)
            return
        # Group "out" deals (entry==1) by position id; sum profit
        out_by_pos: Dict[int, Dict[str, Any]] = {}
        for d in deals:
            if d["entry"] != 1:  # 1 == OUT
                continue
            pid = int(d["position_id"])
            if pid not in out_by_pos:
                out_by_pos[pid] = {"profit": 0.0, "price": d["price"], "time": d["time"], "comment": d["comment"]}
            out_by_pos[pid]["profit"] += float(d["profit"])
            # Use the latest deal's price/time
            if d["time"] > out_by_pos[pid]["time"]:
                out_by_pos[pid]["price"] = d["price"]
                out_by_pos[pid]["time"] = d["time"]
                out_by_pos[pid]["comment"] = d["comment"]

        open_tickets = set(self.journal.open_tickets())
        for pid, info in out_by_pos.items():
            if pid not in open_tickets:
                continue
            close_reason = _classify_close_comment(info["comment"])
            self.journal.record_close(
                ticket=pid,
                closed_at=info["time"],
                close_price=float(info["price"]),
                close_reason=close_reason,
                pnl=float(info["profit"]),
            )
            log.info(
                "Closed ticket=%s reason=%s pnl=%.2f price=%.5f",
                pid, close_reason, info["profit"], info["price"],
            )
        self._last_deals_check = datetime.now(tz=timezone.utc)

    def _resolve_paper_positions(self) -> None:
        """Walk paper positions; close any whose SL/TP was crossed by recent bars."""
        if not self._paper_positions:
            return
        # Pull a small window of recent bars
        df = self.client.get_rates(self.symbol, self.timeframe, 5)
        if df.empty:
            return
        for ticket in list(self._paper_positions.keys()):
            pos = self._paper_positions[ticket]
            relevant = df[df.index > pd.Timestamp(pos["opened_at"]).tz_convert("UTC")]
            if relevant.empty:
                continue
            hit_price: Optional[float] = None
            close_reason = ""
            for _, bar in relevant.iterrows():
                if pos["side"] == "buy":
                    if bar["low"] <= pos["sl"]:
                        hit_price, close_reason = pos["sl"], "sl"
                        break
                    if bar["high"] >= pos["tp"]:
                        hit_price, close_reason = pos["tp"], "tp"
                        break
                else:
                    if bar["high"] >= pos["sl"]:
                        hit_price, close_reason = pos["sl"], "sl"
                        break
                    if bar["low"] <= pos["tp"]:
                        hit_price, close_reason = pos["tp"], "tp"
                        break
            if hit_price is None:
                continue
            pnl_price = (hit_price - pos["entry"]) if pos["side"] == "buy" else (pos["entry"] - hit_price)
            # Rough $ pnl estimate using contract size
            try:
                sym_info = self.client.symbol_info(self.symbol)
                contract = float(getattr(sym_info, "trade_contract_size", 1.0)) or 1.0
            except Exception:  # noqa: BLE001
                contract = 1.0
            pnl_money = pnl_price * pos["volume"] * contract
            self.journal.record_close(
                ticket=ticket,
                closed_at=datetime.now(tz=timezone.utc),
                close_price=float(hit_price),
                close_reason=close_reason,
                pnl=float(pnl_money),
            )
            log.info(
                "[paper] closed ticket=%s reason=%s pnl_price=%.5f pnl=%.2f",
                ticket, close_reason, pnl_price, pnl_money,
            )
            self._paper_positions.pop(ticket, None)


def _classify_close_comment(comment: str) -> str:
    c = (comment or "").lower()
    if "tp" in c or "take" in c:
        return "tp"
    if "sl" in c or "stop" in c:
        return "sl"
    if "robot-close" in c:
        return "manual"
    return "unknown"


# ---------------------------------------------------------------------- entrypoint
def _install_signal_handlers(bot: LiveBot) -> None:
    def _handler(signum, frame):  # noqa: ANN001
        log.info("Received signal %s, stopping...", signum)
        bot.stop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            os_signal.signal(sig, _handler)
        except Exception:  # noqa: BLE001
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Robot Trading live loop")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    bot = LiveBot(cfg)
    _install_signal_handlers(bot)
    bot.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
