"""Pure-pandas indicators and feature engineering.

No TA-Lib dependency on purpose. Keeps install painless on Windows.
All functions take a ``close`` / OHLC DataFrame and return aligned Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder's smoothing == EMA with alpha = 1/period
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def add_indicators(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    ema_trend: int,
    rsi_period: int,
    atr_period: int,
) -> pd.DataFrame:
    """Return a copy of ``df`` with indicator columns appended."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["ema_trend"] = ema(out["close"], ema_trend)
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out, atr_period)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the ML filter.

    Features are designed to be:
      * scale-invariant (ratios / normalised distances), so a model trained on
        XAUUSD doesn't fall apart if price drifts a few hundred dollars
      * causal (no future leakage)
    Assumes ``add_indicators`` has already been called.
    """
    out = pd.DataFrame(index=df.index)

    # Returns over various horizons
    for n in (1, 3, 5, 10, 20):
        out[f"ret_{n}"] = df["close"].pct_change(n)

    # Indicator-based features
    out["ema_fast_slope"] = df["ema_fast"].pct_change(3)
    out["ema_slow_slope"] = df["ema_slow"].pct_change(5)
    out["ema_fast_minus_slow"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]
    out["price_minus_trend"] = (df["close"] - df["ema_trend"]) / df["close"]
    out["rsi"] = df["rsi"] / 100.0
    out["rsi_chg"] = df["rsi"].diff() / 100.0

    # Volatility features
    out["atr_pct"] = df["atr"] / df["close"]
    out["range_pct"] = (df["high"] - df["low"]) / df["close"]
    out["body_pct"] = (df["close"] - df["open"]) / df["close"]

    # Wick ratio (rejection candles tend to mark turning points; lets the model
    # learn whether it's entering on a clean breakout or chasing a wick).
    body_abs = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    out["upper_wick_ratio"] = upper_wick / rng
    out["lower_wick_ratio"] = lower_wick / rng
    out["body_to_range"] = body_abs / rng

    # Distance from recent extremes (40-bar window). Trades launched right at
    # local highs / lows behave differently from mid-range trades.
    look = 40
    rolling_high = df["high"].rolling(look, min_periods=10).max()
    rolling_low = df["low"].rolling(look, min_periods=10).min()
    out["dist_to_high_pct"] = (rolling_high - df["close"]) / df["close"]
    out["dist_to_low_pct"]  = (df["close"] - rolling_low) / df["close"]

    # Volume features (tick volume on FX, but still useful)
    vol = df["volume"].astype(float)
    vol_ma = vol.rolling(20, min_periods=5).mean()
    out["vol_ratio"] = vol / vol_ma.replace(0.0, np.nan)

    # Time-of-day features (London / NY sessions matter for XAUUSD)
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour + df.index.minute / 60.0
        out["tod_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["tod_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow"] = df.index.dayofweek.astype(float) / 6.0
        # Major FX session flags (UTC). One-hot keeps the model from having to
        # decode the cyclic encoding for the use-cases that matter most.
        h = df.index.hour
        out["sess_asia"]   = ((h >= 0)  & (h < 8)).astype(float)
        out["sess_london"] = ((h >= 7)  & (h < 16)).astype(float)
        out["sess_ny"]     = ((h >= 12) & (h < 21)).astype(float)
        out["sess_overlap_lon_ny"] = ((h >= 12) & (h < 16)).astype(float)

    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "ema_fast_slope", "ema_slow_slope", "ema_fast_minus_slow", "price_minus_trend",
    "rsi", "rsi_chg",
    "atr_pct", "range_pct", "body_pct",
    "upper_wick_ratio", "lower_wick_ratio", "body_to_range",
    "dist_to_high_pct", "dist_to_low_pct",
    "vol_ratio",
    "tod_sin", "tod_cos", "dow",
    "sess_asia", "sess_london", "sess_ny", "sess_overlap_lon_ny",
]
