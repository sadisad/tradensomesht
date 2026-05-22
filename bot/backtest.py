"""Vectorised backtester for the EMA+RSI+ATR strategy.

Single-position, exit-on-SL-or-TP, no pyramiding. Good enough to validate that
the strategy has any edge before risking demo capital. Not a substitute for
walk-forward / Monte-Carlo analysis, but a sane first filter.

Bar-level fill assumptions (intentionally pessimistic):
  * Entry: at next bar's open after a signal fires on the closed bar
  * If the next bar's range crosses both SL and TP, assume SL was hit (worst case)
  * Spread is applied as a fixed cost in price units per side
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import build_features
from .strategy import EmaRsiAtrStrategy


@dataclass
class BacktestTrade:
    bar_in: pd.Timestamp
    bar_out: pd.Timestamp
    side: str
    entry: float
    sl: float
    tp: float
    exit: float
    bars_held: int
    pnl_price: float        # raw price move per 1 unit (entry-to-exit, sign-aware)
    outcome: int            # 1 win, 0 loss
    reason: str
    features: Dict[str, float]


@dataclass
class BacktestReport:
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_R: float            # average outcome in R-multiples (TP=R*, SL=-1R)
    total_R: float
    max_consec_losses: int
    long_trades: int
    short_trades: int
    trades: List[BacktestTrade] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"trades={self.n_trades} win_rate={self.win_rate:.1%} "
            f"avg_R={self.avg_R:.3f} total_R={self.total_R:.2f} "
            f"max_consec_losses={self.max_consec_losses} "
            f"long={self.long_trades} short={self.short_trades}"
        )


def run_backtest(
    df: pd.DataFrame,
    strategy_cfg: Dict[str, Any],
    risk_cfg: Dict[str, Any],
    spread_price: float = 0.0,
) -> BacktestReport:
    """Run the strategy bar by bar over ``df`` (must contain OHLCV)."""
    strat = EmaRsiAtrStrategy(strategy_cfg, risk_cfg)
    ind = strat.prepare(df)
    feats = build_features(ind)

    sl_mult = float(risk_cfg["atr_sl_mult"])
    tp_mult = float(risk_cfg["atr_tp_mult"])
    rr = tp_mult / sl_mult  # reward per risk

    trades: List[BacktestTrade] = []
    in_pos = False
    pos: Optional[Dict[str, Any]] = None

    # We iterate over closed bars; signal generated on bar i, fill on bar i+1 open
    closes = ind["close"].values
    highs = ind["high"].values
    lows = ind["low"].values
    opens = ind["open"].values
    times = ind.index

    # Pre-compute signal side for every bar using the strategy's evaluate logic
    # (cheaper than calling evaluate per row -- we replicate the rules vectorised)
    long_trend = ind["close"] > ind["ema_trend"]
    short_trend = ind["close"] < ind["ema_trend"]
    long_mom = ind["ema_fast"] > ind["ema_slow"]
    short_mom = ind["ema_fast"] < ind["ema_slow"]
    bull_cross = (ind["ema_fast"].shift(1) <= ind["ema_slow"].shift(1)) & long_mom
    bear_cross = (ind["ema_fast"].shift(1) >= ind["ema_slow"].shift(1)) & short_mom
    long_pullback = long_trend & long_mom & (ind["low"] <= ind["ema_slow"]) & (ind["ema_slow"] <= ind["close"])
    short_pullback = short_trend & short_mom & (ind["high"] >= ind["ema_slow"]) & (ind["ema_slow"] >= ind["close"])
    rsi_long_ok = (ind["rsi"] > strategy_cfg["rsi_long_min"]) & (ind["rsi"] < strategy_cfg["rsi_long_max"])
    rsi_short_ok = (ind["rsi"] > strategy_cfg["rsi_short_min"]) & (ind["rsi"] < strategy_cfg["rsi_short_max"])

    sig_long = (long_trend & (bull_cross | long_pullback) & rsi_long_ok).values
    sig_short = (short_trend & (bear_cross | short_pullback) & rsi_short_ok).values

    atr_vals = ind["atr"].values

    for i in range(len(ind) - 1):
        # 1. Manage existing position first using bar i+1's high/low
        if in_pos and pos is not None:
            j = i  # while looping bar i is "current"; we evaluate fills on this bar
            hi, lo = highs[j], lows[j]
            entry, sl, tp, side = pos["entry"], pos["sl"], pos["tp"], pos["side"]
            hit_sl = (lo <= sl) if side == "buy" else (hi >= sl)
            hit_tp = (hi >= tp) if side == "buy" else (lo <= tp)
            exit_price: Optional[float] = None
            outcome = 0
            if hit_sl and hit_tp:
                exit_price = sl
                outcome = 0
            elif hit_sl:
                exit_price = sl
                outcome = 0
            elif hit_tp:
                exit_price = tp
                outcome = 1
            if exit_price is not None:
                pnl = (exit_price - entry) if side == "buy" else (entry - exit_price)
                trades.append(
                    BacktestTrade(
                        bar_in=pos["bar_in"],
                        bar_out=times[j],
                        side=side,
                        entry=entry, sl=sl, tp=tp,
                        exit=float(exit_price),
                        bars_held=int(j - pos["i_in"]),
                        pnl_price=float(pnl),
                        outcome=int(outcome),
                        reason=pos["reason"],
                        features=pos["features"],
                    )
                )
                in_pos = False
                pos = None

        # 2. New entry on bar i+1's open if a signal fired on bar i and we're flat
        if not in_pos:
            atr_v = atr_vals[i]
            if np.isnan(atr_v) or atr_v <= 0:
                continue
            side: Optional[str] = None
            reason = ""
            if sig_long[i]:
                side, reason = "buy", "long_setup"
            elif sig_short[i]:
                side, reason = "sell", "short_setup"
            if side is None:
                continue
            # Fill on next bar open
            entry = opens[i + 1] + (spread_price if side == "buy" else -spread_price)
            sl = entry - sl_mult * atr_v if side == "buy" else entry + sl_mult * atr_v
            tp = entry + tp_mult * atr_v if side == "buy" else entry - tp_mult * atr_v
            f = {c: feats[c].iloc[i] for c in feats.columns if c in feats}
            f["__side_buy"] = 1.0 if side == "buy" else 0.0
            pos = {
                "side": side,
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "bar_in": times[i + 1],
                "i_in": i + 1,
                "reason": reason,
                "features": f,
            }
            in_pos = True

    # Stats
    n = len(trades)
    wins = sum(t.outcome for t in trades)
    losses = n - wins
    win_rate = (wins / n) if n else 0.0
    # R-multiple: TP = +rr, SL = -1
    Rs = [rr if t.outcome == 1 else -1.0 for t in trades]
    avg_R = float(np.mean(Rs)) if Rs else 0.0
    total_R = float(np.sum(Rs)) if Rs else 0.0
    max_cl = 0
    cur_cl = 0
    for t in trades:
        if t.outcome == 0:
            cur_cl += 1
            max_cl = max(max_cl, cur_cl)
        else:
            cur_cl = 0
    long_n = sum(1 for t in trades if t.side == "buy")
    short_n = n - long_n
    return BacktestReport(
        n_trades=n,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_R=avg_R,
        total_R=total_R,
        max_consec_losses=max_cl,
        long_trades=long_n,
        short_trades=short_n,
        trades=trades,
    )
