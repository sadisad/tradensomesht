"""In-flight trade management.

Walks every open position once per cycle and applies three rules in order:

  1. **Break-even**: once unrealised gain in R-multiples >= ``breakeven_r``,
     pull SL to entry (+/- a small offset on the side of the trade).
  2. **Trailing stop**: once gain >= ``trail_start_r``, ratchet SL behind price
     by ``trail_distance_atr`` * ATR. Never moves SL backwards.
  3. **Time stop**: if a position has been open for more than ``max_bars``
     bars without resolving, force-close at market.

R is computed from the *original* SL distance recorded at trade open; we get it
from the journal (via ``Journal.open_position_dict``) so restarts are fine.

For paper mode the SL/TP changes are written into the in-memory dict (and
persisted by the live loop). For demo mode we send ``TRADE_ACTION_SLTP`` via
``MT5Client.modify_position_sltp`` and leave the actual exit to the broker.
Time-stop closes go through the appropriate close path.

All behavior is gated by config flags: passing an empty/zero config makes every
rule a no-op, so existing trades are unaffected if you turn this module off.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ManagementAction:
    ticket: int
    kind: str            # "breakeven", "trail", "time_stop"
    new_sl: Optional[float]
    note: str


@dataclass
class ManagedPosition:
    """Subset of position state needed for in-flight management.

    ``original_sl_distance`` is the SL distance set at trade open (price units).
    Used to compute R-multiples even after we've moved SL with break-even/trail.
    """
    ticket: int
    side: str
    entry: float
    current_sl: float
    current_tp: float
    original_sl_distance: float
    atr: float
    opened_at: datetime
    bars_open: int = 0


class TradeManager:
    def __init__(self, mgmt_cfg: Dict[str, Any]):
        self.cfg = mgmt_cfg or {}
        self.enabled: bool = bool(self.cfg.get("enabled", False))
        self.breakeven_r: float = float(self.cfg.get("breakeven_r", 0) or 0)
        self.breakeven_offset_atr: float = float(self.cfg.get("breakeven_offset_atr", 0.05) or 0.0)
        self.trail_start_r: float = float(self.cfg.get("trail_start_r", 0) or 0)
        self.trail_distance_atr: float = float(self.cfg.get("trail_distance_atr", 0) or 0)
        self.max_bars: int = int(self.cfg.get("max_bars", 0) or 0)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        position: ManagedPosition,
        current_price: float,
    ) -> Optional[ManagementAction]:
        """Return a single action to apply, or None if nothing to do.

        Order of priority: time_stop > trailing/breakeven (whichever is tighter).
        """
        if not self.enabled:
            return None
        if position.original_sl_distance <= 0:
            return None

        # --- time stop wins regardless of price action
        if self.max_bars > 0 and position.bars_open >= self.max_bars:
            return ManagementAction(
                ticket=position.ticket,
                kind="time_stop",
                new_sl=None,
                note=f"open {position.bars_open} bars >= max {self.max_bars}",
            )

        # --- compute current R-multiple of unrealised gain
        if position.side == "buy":
            gain = current_price - position.entry
        else:
            gain = position.entry - current_price
        r_mult = gain / position.original_sl_distance

        proposed_sl: Optional[float] = None
        kind: Optional[str] = None
        note = ""

        # Trailing stop (takes priority over break-even once activated; the trail
        # SL will already be >= break-even SL by construction)
        if self.trail_start_r > 0 and self.trail_distance_atr > 0 and r_mult >= self.trail_start_r:
            trail_dist = self.trail_distance_atr * position.atr
            if position.side == "buy":
                candidate = current_price - trail_dist
                if candidate > position.current_sl:
                    proposed_sl = candidate
                    kind = "trail"
                    note = f"r_mult={r_mult:.2f} trail_dist={trail_dist:.5f}"
            else:
                candidate = current_price + trail_dist
                if candidate < position.current_sl:
                    proposed_sl = candidate
                    kind = "trail"
                    note = f"r_mult={r_mult:.2f} trail_dist={trail_dist:.5f}"

        # Break-even (only if trail hasn't already moved us past it)
        if proposed_sl is None and self.breakeven_r > 0 and r_mult >= self.breakeven_r:
            offset = self.breakeven_offset_atr * position.atr
            if position.side == "buy":
                candidate = position.entry + offset
                if candidate > position.current_sl:
                    proposed_sl = candidate
                    kind = "breakeven"
                    note = f"r_mult={r_mult:.2f} offset_atr={self.breakeven_offset_atr}"
            else:
                candidate = position.entry - offset
                if candidate < position.current_sl:
                    proposed_sl = candidate
                    kind = "breakeven"
                    note = f"r_mult={r_mult:.2f} offset_atr={self.breakeven_offset_atr}"

        if proposed_sl is None or kind is None:
            return None
        return ManagementAction(
            ticket=position.ticket,
            kind=kind,
            new_sl=round(float(proposed_sl), 5),
            note=note,
        )
