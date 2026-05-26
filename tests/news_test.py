"""Smoke tests for the news aggregator. Run via:

    python -m tests.news_test

These don't hit the network -- they exercise the RSS parser and tagger with
hand-crafted XML so we don't depend on upstream feed availability.
"""
from __future__ import annotations

from bot.dashboard import _classify_tags, _parse_rss, _strip_html


def main() -> int:
    sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Gold prices climb on Fed rate cut bets</title>
      <link>https://example.com/a</link>
      <description>Spot &lt;b&gt;gold&lt;/b&gt; rose 1.2% as the dollar weakened ahead of the FOMC.</description>
      <pubDate>Mon, 26 May 2025 04:00:00 GMT</pubDate>
    </item>
    <item>
      <title>EUR/USD drifts ahead of ECB decision</title>
      <link>https://example.com/b</link>
      <description>The euro held steady ahead of the central bank meeting.</description>
      <pubDate>Mon, 26 May 2025 03:30:00 GMT</pubDate>
    </item>
    <item>
      <title>Tech stocks rally on chip demand</title>
      <link>https://example.com/c</link>
      <description>Semiconductor names led broader equities higher.</description>
      <pubDate>Mon, 26 May 2025 02:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

    items = _parse_rss(sample_rss, {"source": "TEST", "tags": []})
    assert len(items) == 3, items
    assert items[0]["title"].startswith("Gold prices"), items[0]
    assert "<b>" not in items[0]["summary"], "html stripping failed"
    assert items[0]["link"] == "https://example.com/a"
    assert items[0]["published_at"] is not None

    # Auto-tagging from keywords
    assert "gold" in items[0]["tags"], items[0]["tags"]
    assert "forex" in items[1]["tags"], items[1]["tags"]
    assert "gold" not in items[2]["tags"], items[2]["tags"]
    assert "forex" not in items[2]["tags"], items[2]["tags"]
    print(f"[OK] _parse_rss + _classify_tags produced {len(items)} items with correct tagging")

    # Direct strip / classify checks
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert "gold" in _classify_tags("Bullion at fresh highs", "", [])
    assert "forex" in _classify_tags("USD weakens vs JPY", "", [])
    assert "forex" in _classify_tags("Fed signals rate cut", "", [])
    print("[OK] keyword classifier covers expected tokens")

    print("\nAll news tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
