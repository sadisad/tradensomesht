"""Risk manager.

Computes:
  * stop-loss / take-profit prices from ATR
  * position size from a fixed % of equity at risk per trade
  * dynamic risk scaling based on recent win-rate
  * pre-trade safety checks (daily loss limit, cooldowns, max open positions)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class TradePlan:
    side: str
    entry: float
    sl: float
    tp: float
    volume: float
    risk_money: float       # currency at risk between entry and SL
    risk_pct: float         # actual risk % of equity used
    sl_distance: float      # absolute price distance from entry to SL


class RiskRejected(Exception):
    """Raised when a trade is blocked by a risk rule."""


class RiskManager:
    def __init__(self, risk_cfg: Dict[str, Any], trading_cfg: Dict[str, Any]):
        self.r = risk_cfg
        self.t = trading_cfg

    # ------------------------------------------------------------------ planning
    def build_plan(
        self,
        side: str,
        entry: float,
        atr_value: float,
        equity: float,
        symbol_info: Any,
        risk_scale: float = 1.0,
    ) -> TradePlan:
        """Compute SL/TP/volume for a candidate trade.

        ``symbol_info`` is the MT5 symbol info object. We use:
          point, trade_tick_size, trade_tick_value, volume_min, volume_max, volume_step,
          trade_contract_size.
        """
        if atr_value <= 0:
            raise RiskRejected("ATR is zero/negative; cannot size trade")

        sl_dist = float(atr_value) * float(self.r["atr_sl_mult"])
        tp_dist = float(atr_value) * float(self.r["atr_tp_mult"])

        point = float(getattr(symbol_info, "point", 0.0)) or 0.01
        min_stop_pts = float(self.r.get("min_stop_points", 0)) * point
        if sl_dist < min_stop_pts:
            sl_dist = min_stop_pts
            tp_dist = max(tp_dist, sl_dist * (self.r["atr_tp_mult"] / self.r["atr_sl_mult"]))

        if side == "buy":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        # Money risked per 1 lot for this SL distance:
        tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0)) or point
        tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0))
        if tick_size <= 0 or tick_value <= 0:
            # Fallback: contract_size * sl_dist for a rough estimate (USD-quoted symbols)
            contract = float(getattr(symbol_info, "trade_contract_size", 1.0)) or 1.0
            money_per_lot = sl_dist * contract
        else:
            money_per_lot = (sl_dist / tick_size) * tick_value

        if money_per_lot <= 0:
            raise RiskRejected("Could not estimate money-per-lot; check symbol specs")

        risk_pct = float(self.r["risk_per_trade_pct"]) * float(risk_scale)
        risk_pct = max(0.05, min(risk_pct, float(self.r["risk_per_trade_pct"]) * 2.0))
        risk_money = equity * (risk_pct / 100.0)
        raw_volume = risk_money / money_per_lot

        # Round to broker's volume step
        vol_min = float(getattr(symbol_info, "volume_min", 0.01)) or 0.01
        vol_max = float(getattr(symbol_info, "volume_max", 100.0)) or 100.0
        vol_step = float(getattr(symbol_info, "volume_step", 0.01)) or 0.01
        steps = max(1, int(raw_volume / vol_step))
        volume = steps * vol_step
        volume = max(vol_min, min(volume, vol_max))
        volume = max(float(self.r["min_lot"]), min(volume, float(self.r["max_lot"])))
        volume = round(volume, 2)

        return TradePlan(
            side=side,
            entry=float(entry),
            sl=round(float(sl), 5),
            tp=round(float(tp), 5),
            volume=float(volume),
            risk_money=float(volume * money_per_lot),
            risk_pct=float(risk_pct),
            sl_distance=float(sl_dist),
        )

    # ------------------------------------------------------------------ pre-trade gates
    def within_trading_hours(self, now_utc: Optional[datetime] = None) -> bool:
        hours = self.t.get("trading_hours_utc") or {}
        start_s = hours.get("start")
        end_s = hours.get("end")
        if not start_s or not end_s:
            return True
        now_utc = now_utc or datetime.now(tz=timezone.utc)
        start = time.fromisoformat(start_s)
        end = time.fromisoformat(end_s)
        cur = now_utc.time()
        if start <= end:
            return start <= cur <= end
        # Wraps midnight
        return cur >= start or cur <= end

    def cooldown_active(
        self,
        last_loss_time_utc: Optional[datetime],
        now_utc: Optional[datetime] = None,
    ) -> bool:
        cd = int(self.t.get("cooldown_minutes_after_loss", 0) or 0)
        if cd <= 0 or last_loss_time_utc is None:
            return False
        now_utc = now_utc or datetime.now(tz=timezone.utc)
        return now_utc - last_loss_time_utc < timedelta(minutes=cd)

    def daily_loss_breached(
        self,
        start_of_day_equity: float,
        current_equity: float,
    ) -> bool:
        limit = float(self.r.get("daily_max_loss_pct", 0) or 0)
        if limit <= 0 or start_of_day_equity <= 0:
            return False
        drop_pct = (start_of_day_equity - current_equity) / start_of_day_equity * 100.0
        return drop_pct >= limit

    # ------------------------------------------------------------------ adaptation
    def risk_scale_from_history(self, recent_outcomes: List[int]) -> float:
        """Scale risk between 0.5x and 1.5x based on rolling win-rate.

        ``recent_outcomes`` is a list of 1 (win) / 0 (loss) for the last N closed
        trades of this setup. With <10 samples we don't scale (return 1.0).
        """
        n = len(recent_outcomes)
        if n < 10:
            return 1.0
        wr = sum(recent_outcomes) / n
        # Map win-rate 0.30 -> 0.5x, 0.50 -> 1.0x, 0.70 -> 1.5x (clamped)
        scale = 0.5 + (wr - 0.30) * 2.5
        return max(0.5, min(1.5, scale))
