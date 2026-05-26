"""Smoke tests for the calendar classifier and mixed-data detector.

No network. Run via:

    python -m tests.calendar_test
"""
from __future__ import annotations

from bot.dashboard import (
    _classify_outcome,
    _detect_mixed,
    _enrich_event,
    _intervention_warnings,
    _parse_ff_number,
)


def test_parse_numbers():
    assert _parse_ff_number("3.2%") == 3.2
    assert _parse_ff_number("250K") == 250_000.0
    assert _parse_ff_number("-1.5B") == -1_500_000_000.0
    assert _parse_ff_number(",") is None
    assert _parse_ff_number("--") is None
    assert _parse_ff_number(None) is None
    print("[OK] _parse_ff_number")


def test_classify_higher_is_hawkish():
    # CPI: actual > forecast => beat / hawkish
    out = _classify_outcome("US CPI y/y", forecast=3.2, previous=3.1, actual=3.5)
    assert out["outcome"] == "beat", out
    assert out["bias"] == "hawkish", out
    # Same release missing => dovish
    out = _classify_outcome("US CPI y/y", forecast=3.2, previous=3.1, actual=2.8)
    assert out["outcome"] == "miss", out
    assert out["bias"] == "dovish", out
    print("[OK] CPI beat/miss bias")


def test_classify_lower_is_hawkish():
    # Unemployment rate: lower actual => beat / hawkish
    out = _classify_outcome("Unemployment Rate", forecast=4.0, previous=4.1, actual=3.7)
    assert out["outcome"] == "beat", out
    assert out["bias"] == "hawkish", out
    # Higher unemployment => miss / dovish
    out = _classify_outcome("Unemployment Rate", forecast=4.0, previous=4.1, actual=4.4)
    assert out["outcome"] == "miss", out
    assert out["bias"] == "dovish", out
    print("[OK] Unemployment direction inverted")


def test_classify_inline_band():
    # Within 0.5% tolerance band => inline
    out = _classify_outcome("Retail Sales m/m", forecast=0.5, previous=0.4, actual=0.5)
    assert out["outcome"] == "inline", out
    print("[OK] inline band")


def test_pre_release_expectation():
    out = _classify_outcome("US CPI y/y", forecast=3.5, previous=3.1, actual=None)
    assert out["outcome"] == "pending", out
    assert out["bias"] == "hawkish", out  # higher forecast vs prev = hawkish lean
    print("[OK] pre-release expectation bias")


def test_mixed_detection():
    events = [
        {"currency": "USD", "ts": 1000.0, "title": "Employment Change",
         "outcome": "beat", "bias": "hawkish", "time": "t1"},
        {"currency": "USD", "ts": 1100.0, "title": "Unemployment Rate",
         "outcome": "miss", "bias": "dovish", "time": "t2"},
        {"currency": "USD", "ts": 100000.0, "title": "Retail Sales",
         "outcome": "beat", "bias": "hawkish", "time": "t3"},
    ]
    _detect_mixed(events, window_minutes=30)
    assert events[0].get("mixed"), events[0]
    assert events[1].get("mixed"), events[1]
    assert not events[2].get("mixed"), events[2]
    print("[OK] mixed-data detection")


def test_enrich_event():
    raw = {
        "title": "CPI y/y",
        "country": "USD",
        "date": "2026-05-26T08:30:00-04:00",
        "impact": "High",
        "forecast": "3.2%",
        "previous": "3.1%",
        "actual": "3.5%",
    }
    e = _enrich_event(raw)
    assert e is not None
    assert e["currency"] == "USD"
    assert e["impact"] == "High"
    assert e["outcome"] == "beat"
    assert e["bias"] == "hawkish"
    print("[OK] _enrich_event end-to-end")


def test_intervention_warnings():
    assert _intervention_warnings("USDJPY", 161.0)[0]["level"] == "high"
    assert _intervention_warnings("USDJPY", 156.0)[0]["level"] == "medium"
    assert _intervention_warnings("USDJPY", 150.0) == []
    eurjpy = _intervention_warnings("EURJPY", 170.0)
    assert eurjpy and eurjpy[0]["level"] == "info"
    assert _intervention_warnings("EURUSD", 1.08) == []
    print("[OK] intervention warnings")


def test_priced_in_detection():
    from bot.dashboard import _enrich_event
    # Rate decision where forecast == previous => priced_in
    raw = {
        "title": "RBA Cash Rate",
        "country": "AUD",
        "date": "2026-05-26T04:30:00+00:00",
        "impact": "High",
        "forecast": "4.35%",
        "previous": "4.35%",
        "actual": None,
    }
    e = _enrich_event(raw)
    assert e is not None
    assert e["is_rate_decision"] is True, e
    assert e["priced_in"] is True, e
    # Same release with different numbers => not priced in
    raw["forecast"] = "4.10%"
    e2 = _enrich_event(raw)
    assert e2["is_rate_decision"] is True
    assert e2["priced_in"] is False
    # Non-rate event should not flag rate-decision regardless of equality
    raw_cpi = {
        "title": "CPI y/y",
        "country": "USD",
        "date": "2026-05-26T12:30:00+00:00",
        "impact": "High",
        "forecast": "3.0%",
        "previous": "3.0%",
        "actual": None,
    }
    e3 = _enrich_event(raw_cpi)
    assert e3["is_rate_decision"] is False
    assert e3["priced_in"] is False
    print("[OK] priced-in detection")


def test_statement_tone_scanner():
    from bot.dashboard import _scan_statement_tone
    items = [
        {"title": "RBA holds rates, signals more hikes if inflation persistent",
         "summary": "Bullock said the board remains vigilant on inflation",
         "source": "ForexLive",
         "published_at": "2026-05-26T04:35:00+00:00"},
        {"title": "RBA statement: hawkish tone surprises markets",
         "summary": "Higher for longer guidance",
         "source": "ActionForex",
         "published_at": "2026-05-26T04:40:00+00:00"},
        {"title": "Tech stocks rally", "summary": "unrelated",
         "source": "MarketWatch",
         "published_at": "2026-05-26T04:50:00+00:00"},
    ]
    out = _scan_statement_tone(items, "AUD", "2026-05-26T04:30:00+00:00")
    assert out["tone"] == "hawkish", out
    assert out["hawkish_hits"] >= 2, out
    print("[OK] statement tone scanner")


def main() -> int:
    test_parse_numbers()
    test_classify_higher_is_hawkish()
    test_classify_lower_is_hawkish()
    test_classify_inline_band()
    test_pre_release_expectation()
    test_mixed_detection()
    test_enrich_event()
    test_intervention_warnings()
    test_priced_in_detection()
    test_statement_tone_scanner()
    print("\nAll calendar tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
