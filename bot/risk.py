"""Risk manager.

Computes:
  * stop-loss / take-profit prices from ATR
  * position size from a fixed % of equity at risk per trade
  * dynamic risk scaling based on recent win-rate (linear or fractional Kelly)
  * pre-trade safety checks: trading hours, post-loss cooldown, daily and weekly
    loss limits, max open positions, spread cap, volatility regime
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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

    def weekly_loss_breached(
        self,
        start_of_week_equity: float,
        current_equity: float,
    ) -> bool:
        limit = float(self.r.get("weekly_max_loss_pct", 0) or 0)
        if limit <= 0 or start_of_week_equity <= 0:
            return False
        drop_pct = (start_of_week_equity - current_equity) / start_of_week_equity * 100.0
        return drop_pct >= limit

    def volatility_regime_ok(self, atr_value: float, ref_price: float) -> Tuple[bool, str]:
        """Return (allowed, reason). ATR% outside the [floor, ceiling] band is rejected.

        - Below the floor: market is dead chop, ATR-derived SLs collapse to ``min_stop_points``
          floor and risk:reward becomes random.
        - Above the ceiling: news spike or gap; spreads widen, slippage explodes,
          historical ML doesn't generalize.
        """
        if ref_price <= 0:
            return False, "regime:ref_price<=0"
        atr_pct = float(atr_value) / float(ref_price) * 100.0
        floor = float(self.r.get("regime_atr_pct_min", 0) or 0)
        ceiling = float(self.r.get("regime_atr_pct_max", 0) or 0)
        if floor > 0 and atr_pct < floor:
            return False, f"regime_atr_pct_too_low({atr_pct:.4f}<{floor:.4f})"
        if ceiling > 0 and atr_pct > ceiling:
            return False, f"regime_atr_pct_too_high({atr_pct:.4f}>{ceiling:.4f})"
        return True, ""

    def spread_acceptable(
        self,
        spread_points: float,
        atr_value: float,
        symbol_info: Any,
    ) -> Tuple[bool, str]:
        """Reject when spread is large vs the SL we'd build off ATR, or vs an absolute cap.

        Returns (allowed, reason).
        """
        cap_points = float(self.r.get("max_spread_points", 0) or 0)
        if cap_points > 0 and spread_points > cap_points:
            return False, f"spread_too_wide({spread_points:.1f}pts>{cap_points:.1f}pts)"
        max_ratio = float(self.r.get("max_spread_atr_ratio", 0) or 0)
        if max_ratio > 0 and atr_value > 0:
            point = float(getattr(symbol_info, "point", 0.0)) or 0.00001
            atr_points = float(atr_value) / point
            ratio = spread_points / atr_points if atr_points > 0 else 0.0
            if ratio > max_ratio:
                return False, f"spread_atr_ratio_too_high({ratio:.3f}>{max_ratio:.3f})"
        return True, ""

    # ------------------------------------------------------------------ adaptation
    def risk_scale_from_history(self, recent_outcomes: List[int]) -> float:
        """Scale risk based on rolling win-rate, using either linear blend or
        fractional Kelly depending on ``risk.scaling_mode`` ("linear" or "kelly").

        ``recent_outcomes`` is a list of 1 (win) / 0 (loss) for the last N closed
        trades of this setup. With <``risk.scaling_min_samples`` samples we don't
        scale (return 1.0) so we don't whipsaw on noise.
        """
        n = len(recent_outcomes)
        min_samples = int(self.r.get("scaling_min_samples", 10))
        if n < min_samples:
            return 1.0

        wr = sum(recent_outcomes) / n
        mode = str(self.r.get("scaling_mode", "linear")).lower()

        if mode == "kelly":
            return self._kelly_scale(wr)
        # default: linear win-rate blend
        # Map win-rate 0.30 -> 0.5x, 0.50 -> 1.0x, 0.70 -> 1.5x (clamped)
        scale = 0.5 + (wr - 0.30) * 2.5
        return max(0.5, min(1.5, scale))

    def _kelly_scale(self, win_rate: float) -> float:
        """Fractional-Kelly multiplier on base risk.

        Kelly fraction:  f* = (b*p - q) / b   with b = TP_mult / SL_mult, q = 1-p.
        We normalise by the Kelly fraction at p=0.5 so that "even-money win-rate
        with positive R:R" maps to a 1.0x multiplier. Above 0.5 -> scale up,
        below -> scale down. Then we apply ``kelly_fraction`` (default 0.25 =
        quarter-Kelly) to dampen sensitivity to a noisy win-rate estimate, and
        clamp to [scaling_floor, scaling_ceiling]. Negative-edge -> floor.
        """
        sl_mult = float(self.r.get("atr_sl_mult", 1.0))
        tp_mult = float(self.r.get("atr_tp_mult", 1.0))
        b = (tp_mult / sl_mult) if sl_mult > 0 else 1.0
        if b <= 0:
            return 1.0
        p = float(win_rate)
        kelly = (b * p - (1.0 - p)) / b
        kelly_at_50 = (b * 0.5 - 0.5) / b
        if kelly_at_50 <= 0:
            return 1.0  # negative R:R: don't scale at all, base risk handles it
        ratio = kelly / kelly_at_50
        # Soft fractional-Kelly: blend toward 1.0 by (1 - kelly_fraction).
        # kelly_fraction=1 -> use raw ratio; =0.25 -> mostly stay near 1.0.
        frac = max(0.0, min(1.0, float(self.r.get("kelly_fraction", 0.25))))
        mult = 1.0 + (ratio - 1.0) * frac
        floor = float(self.r.get("scaling_floor", 0.5))
        ceiling = float(self.r.get("scaling_ceiling", 1.5))
        return max(floor, min(ceiling, mult))
