"""Smoke tests for the in-flight TradeManager, especially the news_ride
partial-close rule. No network. Run via:

    python -m tests.trade_management_test
"""
from __future__ import annotations

from datetime import datetime, timezone

from bot.trade_management import ManagedPosition, TradeManager


def _pos(side="buy", entry=2000.0, sl=1990.0, tp=2030.0, vol=0.10,
         atr=10.0, partial_taken=False, bars_open=0):
    return ManagedPosition(
        ticket=1,
        side=side,
        entry=entry,
        current_sl=sl,
        current_tp=tp,
        original_sl_distance=abs(entry - sl),
        original_volume=vol,
        atr=atr,
        opened_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
        bars_open=bars_open,
        partial_taken=partial_taken,
    )


def test_news_ride_disabled_does_nothing():
    tm = TradeManager({"enabled": True, "news_ride": {"enabled": False, "trigger_r": 1.0}})
    a = tm.evaluate(_pos(), current_price=2010.0)  # 1R move
    assert a is None, a
    print("[OK] news_ride disabled is a no-op")


def test_news_ride_fires_at_trigger():
    tm = TradeManager({
        "enabled": True,
        "breakeven_offset_atr": 0.05,
        "news_ride": {"enabled": True, "trigger_r": 1.0, "partial_pct": 0.9},
    })
    p = _pos()  # entry 2000 sl 1990 vol 0.10
    a = tm.evaluate(p, current_price=2010.0)
    assert a is not None
    assert a.kind == "partial_close", a
    assert abs(a.close_volume - 0.09) < 1e-6, a.close_volume
    assert a.set_breakeven is True
    # SL should be entry + 0.05 * ATR = 2000 + 0.5 = 2000.5
    assert abs(a.new_sl - 2000.5) < 1e-3, a.new_sl
    print("[OK] news_ride fires at trigger_r with correct partial size")


def test_news_ride_only_fires_once():
    tm = TradeManager({
        "enabled": True,
        "news_ride": {"enabled": True, "trigger_r": 1.0, "partial_pct": 0.9},
    })
    p = _pos(partial_taken=True)
    a = tm.evaluate(p, current_price=2050.0)
    # Even at 5R, with partial_taken the news_ride branch is skipped. Other
    # rules (BE/trail) might still fire if configured -- here they aren't.
    assert a is None or a.kind != "partial_close", a
    print("[OK] news_ride does not re-fire after partial_taken")


def test_news_ride_below_trigger_does_nothing():
    tm = TradeManager({
        "enabled": True,
        "news_ride": {"enabled": True, "trigger_r": 1.5, "partial_pct": 0.9},
    })
    a = tm.evaluate(_pos(), current_price=2010.0)  # only 1R, need 1.5R
    assert a is None, a
    print("[OK] news_ride respects trigger_r")


def test_sell_side_news_ride():
    tm = TradeManager({
        "enabled": True,
        "breakeven_offset_atr": 0.05,
        "news_ride": {"enabled": True, "trigger_r": 1.0, "partial_pct": 0.9},
    })
    # short from 2000 with SL at 2010, target at 1980
    p = _pos(side="sell", entry=2000.0, sl=2010.0, tp=1980.0)
    a = tm.evaluate(p, current_price=1990.0)  # 1R move down
    assert a is not None and a.kind == "partial_close", a
    # SL should be entry - 0.05 * ATR = 2000 - 0.5 = 1999.5
    assert abs(a.new_sl - 1999.5) < 1e-3, a.new_sl
    print("[OK] sell-side news_ride pulls SL above entry by offset")


def test_breakeven_still_works_after_partial():
    tm = TradeManager({
        "enabled": True,
        "breakeven_r": 1.0,
        "breakeven_offset_atr": 0.05,
        "news_ride": {"enabled": True, "trigger_r": 1.0, "partial_pct": 0.9},
    })
    p = _pos(partial_taken=True)
    # After partial_taken, news_ride is skipped; breakeven should now fire.
    a = tm.evaluate(p, current_price=2010.0)
    assert a is not None and a.kind == "breakeven", a
    print("[OK] breakeven still applies after partial_taken")


def main() -> int:
    test_news_ride_disabled_does_nothing()
    test_news_ride_fires_at_trigger()
    test_news_ride_only_fires_once()
    test_news_ride_below_trigger_does_nothing()
    test_sell_side_news_ride()
    test_breakeven_still_works_after_partial()
    print("\nAll trade-management tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
