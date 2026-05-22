"""EMA + RSI trend-following strategy with ATR-based stops.

Signal rules (long mirrored for short):

  * Trend filter:  close > ema_trend           (uptrend)
  * Momentum:      ema_fast > ema_slow         (fast above slow)
  * Trigger:       previous bar ema_fast <= ema_slow AND current bar ema_fast > ema_slow
                   (i.e. fresh bullish cross), OR a pullback that holds the slow ema
                   while RSI is in the long zone
  * RSI gate:      rsi_long_min < rsi < rsi_long_max  (avoid chasing tops)
  * MTF gate:      (optional) higher-timeframe close above its EMA confirms trend.
                   Enabled via strategy.mtf_enabled. Without confirmation, signals
                   in the opposite direction of the HTF trend are dropped.

The strategy emits at most one signal per bar. The live loop decides whether
to act on it based on risk, ML filter, cooldowns, and existing positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .indicators import add_indicators, ema


@dataclass
class Signal:
    side: str            # "buy" or "sell"
    reason: str          # human-readable description
    entry: float         # last close (used as reference price)
    atr: float           # current ATR value
    bar_time: pd.Timestamp


class EmaRsiAtrStrategy:
    def __init__(self, strategy_cfg: Dict[str, Any], risk_cfg: Dict[str, Any]):
        self.s = strategy_cfg
        self.r = risk_cfg

    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_indicators(
            df,
            ema_fast=int(self.s["ema_fast"]),
            ema_slow=int(self.s["ema_slow"]),
            ema_trend=int(self.s["ema_trend"]),
            rsi_period=int(self.s["rsi_period"]),
            atr_period=int(self.r["atr_period"]),
        )

    # ------------------------------------------------------------------
    def htf_bias(self, htf_df: Optional[pd.DataFrame]) -> Optional[str]:
        """Return "long", "short", or None given a higher-timeframe OHLCV frame.

        Bias rule: HTF close vs HTF EMA(``mtf_ema_period``). Above -> long bias,
        below -> short bias. Returns None if disabled or data is insufficient.
        """
        if htf_df is None or htf_df.empty:
            return None
        period = int(self.s.get("mtf_ema_period", 50))
        if len(htf_df) < period + 2:
            return None
        ema_series = ema(htf_df["close"], period)
        last_close = float(htf_df["close"].iloc[-1])
        last_ema = float(ema_series.iloc[-1])
        if pd.isna(last_ema):
            return None
        if last_close > last_ema:
            return "long"
        if last_close < last_ema:
            return "short"
        return None

    # ------------------------------------------------------------------
    def evaluate(
        self,
        df_with_ind: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        """Return a Signal for the *last closed* bar, or None.

        If ``htf_df`` is provided AND ``strategy.mtf_enabled`` is true, signals
        whose side conflicts with the HTF bias are filtered out.
        """
        if len(df_with_ind) < int(self.s["ema_trend"]) + 5:
            return None

        # Use the last fully-closed bar. In a typical loop we already pulled bars
        # up to "now"; the very last row may still be forming. We treat the
        # second-to-last row as the closed bar to avoid acting on a partial candle.
        prev = df_with_ind.iloc[-3]
        cur = df_with_ind.iloc[-2]

        # Reject rows where indicators haven't warmed up yet
        for col in ("ema_fast", "ema_slow", "ema_trend", "rsi", "atr"):
            if pd.isna(cur[col]) or pd.isna(prev[col]):
                return None

        long_trend = cur["close"] > cur["ema_trend"]
        short_trend = cur["close"] < cur["ema_trend"]

        long_mom = cur["ema_fast"] > cur["ema_slow"]
        short_mom = cur["ema_fast"] < cur["ema_slow"]

        bull_cross = (prev["ema_fast"] <= prev["ema_slow"]) and long_mom
        bear_cross = (prev["ema_fast"] >= prev["ema_slow"]) and short_mom

        # Pullback continuation: price taps slow EMA from the right side and bounces
        long_pullback = (
            long_trend and long_mom and
            cur["low"] <= cur["ema_slow"] <= cur["close"]
        )
        short_pullback = (
            short_trend and short_mom and
            cur["high"] >= cur["ema_slow"] >= cur["close"]
        )

        rsi_long_ok = self.s["rsi_long_min"] < cur["rsi"] < self.s["rsi_long_max"]
        rsi_short_ok = self.s["rsi_short_min"] < cur["rsi"] < self.s["rsi_short_max"]

        mtf_enabled = bool(self.s.get("mtf_enabled", False))
        bias = self.htf_bias(htf_df) if mtf_enabled else None

        if long_trend and (bull_cross or long_pullback) and rsi_long_ok:
            if mtf_enabled and bias == "short":
                return None
            reason = "bull_cross" if bull_cross else "long_pullback"
            if bias == "long":
                reason += "+mtf"
            return Signal(
                side="buy",
                reason=reason,
                entry=float(cur["close"]),
                atr=float(cur["atr"]),
                bar_time=cur.name,
            )

        if short_trend and (bear_cross or short_pullback) and rsi_short_ok:
            if mtf_enabled and bias == "long":
                return None
            reason = "bear_cross" if bear_cross else "short_pullback"
            if bias == "short":
                reason += "+mtf"
            return Signal(
                side="sell",
                reason=reason,
                entry=float(cur["close"]),
                atr=float(cur["atr"]),
                bar_time=cur.name,
            )

        return None
