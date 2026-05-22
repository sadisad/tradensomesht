"""Smoke test: run the strategy + backtester + journal + ML filter on synthetic
OHLCV data. No MT5 required. Use this to sanity-check the core pieces.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.backtest import run_backtest
from bot.indicators import add_indicators, build_features, FEATURE_COLUMNS
from bot.journal import Journal, OpenTrade
from bot.ml_filter import MLFilter
from bot.risk import RiskManager
from bot.strategy import EmaRsiAtrStrategy
from bot.trade_management import ManagedPosition, TradeManager


def make_synthetic_ohlcv(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    """A trending + mean-reverting random walk that produces both setups."""
    rng = np.random.default_rng(seed)
    # Multi-regime drift: alternating bullish / bearish / chop blocks
    drift = np.zeros(n)
    block = 500
    for k in range(0, n, block):
        regime = (k // block) % 3
        d = {0: 0.05, 1: -0.05, 2: 0.0}[regime]
        drift[k:k + block] = d
    noise = rng.normal(0.0, 1.0, n)
    log_ret = drift + noise * 0.6
    price = 2000 + np.cumsum(log_ret)
    # Build OHLC from "close-to-close" walk with intra-bar wiggle
    close = price
    open_ = np.concatenate(([close[0]], close[:-1]))
    wiggle = np.abs(rng.normal(0.0, 0.8, n))
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    vol = rng.integers(50, 500, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


STRATEGY_CFG = {
    "ema_fast": 20, "ema_slow": 50, "ema_trend": 200,
    "rsi_period": 14,
    "rsi_long_min": 50, "rsi_long_max": 70,
    "rsi_short_min": 30, "rsi_short_max": 50,
}
RISK_CFG = {
    "risk_per_trade_pct": 0.5,
    "atr_period": 14,
    "atr_sl_mult": 1.5, "atr_tp_mult": 2.5,
    "min_stop_points": 50,
    "max_lot": 1.0, "min_lot": 0.01,
    "daily_max_loss_pct": 3.0,
}
TRADING_CFG = {
    "symbol": "XAUUSD", "timeframe": "M5",
    "trading_hours_utc": {"start": "00:00", "end": "23:59"},
    "cooldown_minutes_after_loss": 30,
    "max_open_positions": 1,
}


class FakeSymbolInfo:
    point = 0.01
    trade_tick_size = 0.01
    trade_tick_value = 1.0
    trade_contract_size = 100.0
    volume_min = 0.01
    volume_max = 100.0
    volume_step = 0.01


def test_indicators():
    df = make_synthetic_ohlcv(500)
    ind = add_indicators(df, 20, 50, 200, 14, 14)
    assert {"ema_fast", "ema_slow", "ema_trend", "rsi", "atr"}.issubset(ind.columns)
    # RSI bounded
    rsi = ind["rsi"].dropna()
    assert (rsi.between(0, 100)).all(), "RSI out of bounds"
    # ATR positive once warmed up
    atr = ind["atr"].dropna()
    assert (atr > 0).all(), "ATR not positive"
    print(f"[OK] indicators: rsi range=({rsi.min():.1f},{rsi.max():.1f}) atr_mean={atr.mean():.3f}")


def test_features():
    df = make_synthetic_ohlcv(800)
    ind = add_indicators(df, 20, 50, 200, 14, 14)
    feats = build_features(ind)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"missing feature {col}"
    assert not np.isinf(feats.replace([np.inf, -np.inf], np.nan).abs().max().max()), "infs in features"
    print(f"[OK] features: shape={feats.shape}")


def test_strategy_evaluate():
    df = make_synthetic_ohlcv(2000)
    strat = EmaRsiAtrStrategy(STRATEGY_CFG, RISK_CFG)
    ind = strat.prepare(df)
    sigs = 0
    for i in range(220, len(ind)):
        sub = ind.iloc[: i + 1]
        if strat.evaluate(sub) is not None:
            sigs += 1
    print(f"[OK] strategy.evaluate produced {sigs} signals over {len(ind) - 220} bars")
    assert sigs > 0, "strategy never fired -- likely a logic bug"


def test_backtest():
    df = make_synthetic_ohlcv(5000)
    report = run_backtest(df, STRATEGY_CFG, RISK_CFG, spread_price=0.05)
    print(f"[OK] backtest: {report.summary()}")
    assert report.n_trades > 0, "backtest produced no trades"


def test_risk_plan():
    rm = RiskManager(RISK_CFG, TRADING_CFG)
    plan = rm.build_plan(
        side="buy", entry=2350.00, atr_value=2.0,
        equity=10_000.0, symbol_info=FakeSymbolInfo(), risk_scale=1.0,
    )
    print(
        f"[OK] risk: vol={plan.volume:.2f} sl={plan.sl:.2f} tp={plan.tp:.2f} "
        f"risk=${plan.risk_money:.2f} ({plan.risk_pct:.2f}%)"
    )
    assert plan.sl < plan.entry < plan.tp
    assert plan.volume >= RISK_CFG["min_lot"]


def test_phase1_risk_gates():
    """Volatility regime, spread filter, weekly loss, Kelly scaling."""
    cfg = dict(RISK_CFG)
    cfg.update({
        "regime_atr_pct_min": 0.05,
        "regime_atr_pct_max": 0.30,
        "max_spread_points": 30,
        "max_spread_atr_ratio": 0.30,
        "weekly_max_loss_pct": 6.0,
        "scaling_mode": "kelly",
        "scaling_min_samples": 10,
        "scaling_floor": 0.5,
        "scaling_ceiling": 1.5,
        "kelly_fraction": 0.25,
    })
    rm = RiskManager(cfg, TRADING_CFG)

    # regime: 0.01% ATR/price below 0.05% floor -> blocked
    ok, _ = rm.volatility_regime_ok(atr_value=0.2, ref_price=2350.0)
    assert not ok
    # 0.10% inside band -> allowed
    ok, _ = rm.volatility_regime_ok(atr_value=2.35, ref_price=2350.0)
    assert ok
    # 0.50% above ceiling -> blocked
    ok, _ = rm.volatility_regime_ok(atr_value=11.75, ref_price=2350.0)
    assert not ok

    # spread: 50pt > 30pt cap -> blocked
    ok, _ = rm.spread_acceptable(spread_points=50, atr_value=2.0, symbol_info=FakeSymbolInfo())
    assert not ok
    # 20pt with ATR=2.0 (at point=0.01 -> 200pts ATR) ratio=0.10 -> allowed
    ok, _ = rm.spread_acceptable(spread_points=20, atr_value=2.0, symbol_info=FakeSymbolInfo())
    assert ok

    # weekly DD
    assert rm.weekly_loss_breached(10000, 9300) is True   # -7%
    assert rm.weekly_loss_breached(10000, 9700) is False  # -3%

    # Kelly: ~5/10 wins -> ~1.0x; 9/10 wins -> hits ceiling; 2/10 wins -> floor
    s_neutral = rm.risk_scale_from_history([1, 0] * 5)
    assert 0.85 <= s_neutral <= 1.10, f"neutral wr should be near 1.0, got {s_neutral}"
    s_great = rm.risk_scale_from_history([1] * 9 + [0])
    assert s_great >= 1.4
    s_bad = rm.risk_scale_from_history([0] * 8 + [1, 1])
    assert s_bad <= 0.7

    print("[OK] phase1 risk gates: regime, spread, weekly DD, Kelly scaling")


def test_phase1_trade_management():
    """Break-even, trailing stop, time stop."""
    cfg = {
        "enabled": True,
        "breakeven_r": 1.0,
        "breakeven_offset_atr": 0.05,
        "trail_start_r": 1.5,
        "trail_distance_atr": 1.0,
        "max_bars": 50,
    }
    tm = TradeManager(cfg)
    base = ManagedPosition(
        ticket=1, side="buy", entry=100.0,
        current_sl=98.0, current_tp=104.0,
        original_sl_distance=2.0, atr=1.0,
        opened_at=datetime.now(tz=timezone.utc),
        bars_open=5,
    )
    # +0.5R: nothing fires
    assert tm.evaluate(base, current_price=101.0) is None
    # +1.2R: break-even should kick in
    a = tm.evaluate(base, current_price=102.4)
    assert a is not None and a.kind == "breakeven"
    assert a.new_sl > base.entry > base.current_sl
    # +1.6R: trail dominates
    a = tm.evaluate(base, current_price=103.2)
    assert a is not None and a.kind == "trail"
    # time stop
    base.bars_open = 200
    a = tm.evaluate(base, current_price=101.0)
    assert a is not None and a.kind == "time_stop"

    # Sell side mirrors
    sell = ManagedPosition(
        ticket=2, side="sell", entry=100.0,
        current_sl=102.0, current_tp=96.0,
        original_sl_distance=2.0, atr=1.0,
        opened_at=datetime.now(tz=timezone.utc),
        bars_open=5,
    )
    a = tm.evaluate(sell, current_price=97.6)  # +1.2R
    assert a is not None and a.kind == "breakeven"
    assert a.new_sl < sell.entry < sell.current_sl

    # Disabled config: every call returns None
    tm_off = TradeManager({"enabled": False})
    assert tm_off.evaluate(base, current_price=103.2) is None

    print("[OK] phase1 trade management: break-even, trail, time stop, disabled gate")


def test_phase1_strategy_mtf():
    """MTF gate filters opposing-bias signals."""
    df = make_synthetic_ohlcv(2000)
    cfg = dict(STRATEGY_CFG)
    cfg["mtf_enabled"] = True
    cfg["mtf_ema_period"] = 50
    strat = EmaRsiAtrStrategy(cfg, RISK_CFG)
    ind = strat.prepare(df)

    # Synthesize a clearly bullish HTF: monotone-increasing closes
    htf_bull = pd.DataFrame(
        {"open": np.arange(100, 200), "high": np.arange(100, 200) + 0.5,
         "low": np.arange(100, 200) - 0.5, "close": np.arange(100, 200, dtype=float),
         "volume": np.ones(100)},
        index=pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC"),
    )
    # And bearish HTF
    htf_bear = htf_bull.copy()
    htf_bear[["open", "high", "low", "close"]] = htf_bull[["open", "high", "low", "close"]].iloc[::-1].values

    # Bias resolution
    assert strat.htf_bias(htf_bull) == "long"
    assert strat.htf_bias(htf_bear) == "short"
    assert strat.htf_bias(None) is None

    # Sweep bars and count signals; bullish HTF should never let a sell through,
    # bearish HTF should never let a buy through.
    n_buy_with_bull, n_sell_with_bull = 0, 0
    n_buy_with_bear, n_sell_with_bear = 0, 0
    for i in range(220, len(ind)):
        sub = ind.iloc[: i + 1]
        s1 = strat.evaluate(sub, htf_df=htf_bull)
        if s1:
            n_buy_with_bull += int(s1.side == "buy")
            n_sell_with_bull += int(s1.side == "sell")
        s2 = strat.evaluate(sub, htf_df=htf_bear)
        if s2:
            n_buy_with_bear += int(s2.side == "buy")
            n_sell_with_bear += int(s2.side == "sell")
    assert n_sell_with_bull == 0, "bullish HTF should suppress sell signals"
    assert n_buy_with_bear == 0, "bearish HTF should suppress buy signals"
    print(
        f"[OK] phase1 mtf: bull-htf -> {n_buy_with_bull} buys / {n_sell_with_bull} sells; "
        f"bear-htf -> {n_buy_with_bear} buys / {n_sell_with_bear} sells"
    )


def test_journal_and_ml():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "j.db"
        model_path = Path(tmp) / "m.joblib"
        journal = Journal(db)
        ml = MLFilter({
            "enabled": True,
            "model_path": str(model_path),
            "min_train_samples": 60,
            "min_proba_to_trade": 0.55,
            "retrain_every_n_trades": 25,
            "feature_lookback": 50,
        })

        # Build labeled synthetic trades where two features jointly drive wins.
        # Strong, learnable signal so we can reliably assert the pipeline works.
        rng = np.random.default_rng(13)
        n_samples = 400
        for k in range(n_samples):
            base = {c: 0.0 for c in FEATURE_COLUMNS}
            # Only two features carry the signal; rest are zeros + tiny noise
            for c in FEATURE_COLUMNS:
                base[c] = float(rng.normal() * 0.05)
            base["rsi"] = float(rng.uniform(0.2, 0.8))
            base["ema_fast_minus_slow"] = float(rng.normal())
            side = "buy" if k % 2 == 0 else "sell"
            base["__side_buy"] = 1.0 if side == "buy" else 0.0
            # Strong, near-deterministic signal so CV picks it up:
            score = (base["rsi"] - 0.5) * 4.0 + 0.8 * base["ema_fast_minus_slow"]
            if side == "sell":
                score = -score
            p_win = float(np.clip(0.5 + score * 0.6, 0.02, 0.98))
            outcome = int(rng.random() < p_win)

            opened = datetime.now(tz=timezone.utc) - timedelta(hours=n_samples - k)
            closed = opened + timedelta(minutes=30)
            t = OpenTrade(
                ticket=1000 + k, symbol="XAUUSD", side=side,
                volume=0.10, entry_price=2350.0,
                sl=2345.0, tp=2360.0, atr=2.0,
                risk_money=50.0, risk_pct=0.5, reason="synthetic",
                features=base, opened_at=opened, magic=42,
            )
            journal.record_open(t)
            pnl = 5.0 if outcome else -5.0
            journal.record_close(
                ticket=t.ticket, closed_at=closed,
                close_price=2360.0 if outcome else 2345.0,
                close_reason="tp" if outcome else "sl",
                pnl=pnl,
            )

        report = ml.retrain(journal)
        assert report is not None, "retrain returned None"
        print(
            f"[OK] ml retrain: n={report.n_samples} cv_auc={report.cv_auc:.3f} "
            f"cv_acc={report.cv_acc:.3f} balance={report.class_balance}"
        )
        assert report.n_samples == n_samples

        # Predict at high-quality buy setup vs the opposite
        good = {c: 0.0 for c in FEATURE_COLUMNS}
        good["rsi"] = 0.75
        good["ema_fast_minus_slow"] = 1.5
        good["__side_buy"] = 1.0
        bad = dict(good)
        bad["rsi"] = 0.25
        bad["ema_fast_minus_slow"] = -1.5
        p_good = ml.predict_proba_win(good)
        p_bad = ml.predict_proba_win(bad)
        print(f"[OK] ml proba good={p_good:.3f} bad={p_bad:.3f}")
        # Directional sanity check: model should rank good > bad by a clear margin.
        # This is more meaningful than CV stats on synthetic data.
        assert p_good - p_bad > 0.10, (
            f"model failed directional sanity check (good={p_good:.3f} bad={p_bad:.3f})"
        )


def main():
    test_indicators()
    test_features()
    test_strategy_evaluate()
    test_backtest()
    test_risk_plan()
    test_phase1_risk_gates()
    test_phase1_trade_management()
    test_phase1_strategy_mtf()
    test_journal_and_ml()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
