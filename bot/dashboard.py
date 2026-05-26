"""FastAPI dashboard for the trading bot.

Serves a single HTML page (``static/index.html``) plus JSON endpoints that read
from the SQLite journal and (optionally) the MT5 terminal.

Designed to run as a *separate process* alongside the live bot. Both share the
same journal file, so the dashboard always reflects the bot's actual state.

Run:

    python -m bot.dashboard

By default it binds 127.0.0.1:8765. Override with ``--host`` / ``--port``.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time as _time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config, project_root
from .logging_setup import get_logger, setup_logging

log = get_logger(__name__)

# Lazy import: only needed when MT5 is reachable. The dashboard still works
# without the terminal (it just hides live price + equity).
try:  # pragma: no cover
    from .broker_mt5 import MT5Client
except Exception:  # noqa: BLE001
    MT5Client = None  # type: ignore


# ---------------------------------------------------------------------- state
class State:
    """Process-wide singletons. Populated in ``create_app``."""

    cfg: Dict[str, Any] = {}
    db_path: Path = Path("data/journal.db")
    mt5_client: Optional[Any] = None
    mt5_connected: bool = False


STATE = State()


# ---------------------------------------------------------------------- helpers
@contextmanager
def _db():
    conn = sqlite3.connect(STATE.db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _try_mt5():
    """Best-effort connect / reconnect. Failures are non-fatal."""
    if MT5Client is None:
        return None
    if STATE.mt5_client is None:
        try:
            STATE.mt5_client = MT5Client(STATE.cfg.get("broker", {}))
            STATE.mt5_client.connect()
            STATE.mt5_connected = True
            log.info("Dashboard connected to MT5")
        except Exception as e:  # noqa: BLE001
            log.warning("MT5 unavailable for dashboard: %s", e)
            STATE.mt5_client = None
            STATE.mt5_connected = False
    return STATE.mt5_client


def _row_to_dict(r: sqlite3.Row) -> Dict[str, Any]:
    return {k: r[k] for k in r.keys()}


def _available_symbols(default: str, seed: Optional[List[str]] = None) -> List[str]:
    """Symbols this dashboard knows about: union of seed list + journal-observed
    symbols + the configured default. Seeds let multi-bot setups force pairs
    into the picker even before any signal has been recorded."""
    syms = set()
    if default:
        syms.add(default)
    if seed:
        for s in seed:
            if s:
                syms.add(s)
    try:
        with _db() as c:
            for r in c.execute("SELECT DISTINCT symbol FROM trades WHERE symbol IS NOT NULL"):
                syms.add(str(r["symbol"]))
            for r in c.execute("SELECT DISTINCT symbol FROM signals WHERE symbol IS NOT NULL"):
                syms.add(str(r["symbol"]))
    except Exception as e:  # noqa: BLE001
        log.debug("symbol discovery failed (journal not ready?): %s", e)
    return sorted(s for s in syms if s)


def _resolve_symbol(requested: Optional[str], default: str) -> str:
    """Sanitize a user-supplied symbol. We accept anything alphanumeric + a few separators
    so brokers' suffixed names (XAUUSD.s, EURUSD-pro) work; reject anything else to avoid
    feeding garbage to MT5 or SQL."""
    if not requested:
        return default
    cleaned = requested.strip()
    if not cleaned or len(cleaned) > 32:
        return default
    if not all(ch.isalnum() or ch in "._-" for ch in cleaned):
        return default
    return cleaned


# ---------------------------------------------------------------------- news
# Curated forex / gold RSS feeds. Each entry is (source label, URL, default tags).
# Only public RSS endpoints -- no API keys required. We fetch, parse, dedup, and
# cache in-memory so a busy dashboard doesn't hammer the upstream sources.
_NEWS_FEEDS: List[Dict[str, Any]] = [
    {
        "source": "ForexLive",
        "url": "https://www.forexlive.com/feed/news",
        "tags": ["forex"],
    },
    {
        "source": "Investing.com - Forex",
        "url": "https://www.investing.com/rss/forex.rss",
        "tags": ["forex"],
    },
    {
        "source": "Investing.com - Commodities",
        "url": "https://www.investing.com/rss/news_11.rss",
        "tags": ["gold", "commodities"],
    },
    {
        "source": "ActionForex",
        "url": "https://www.actionforex.com/feed/",
        "tags": ["forex"],
    },
    {
        "source": "MarketWatch - Market Pulse",
        "url": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "tags": ["markets"],
    },
]

# Keywords used to auto-tag items as gold/forex regardless of feed origin.
_GOLD_KEYWORDS = (
    "gold", "xau", "bullion", "precious metal", "metals", "silver", "kitco",
)
_FOREX_KEYWORDS = (
    "forex", "fx", "dollar", "usd", "eur", "gbp", "jpy", "cad", "aud", "nzd",
    "chf", "yuan", "yen", "pound", "euro", "currency", "currencies", "fed",
    "ecb", "boj", "boe", "rba", "rbnz", "snb", "central bank", "rate hike",
    "rate cut", "interest rate", "cpi", "inflation", "nfp", "non-farm",
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------- calendar
# Economic calendar via ForexFactory's free JSON feed. Used to drive a
# "trade-the-news" workflow per the WelcomeHomeTrading methodology:
#
#   1. Fundamental = WHY  --> show upcoming high-impact events for the active
#      pair so the trader knows *which* releases matter.
#   2. Plan if-then BEFORE the release --> compute a beat / miss bias from
#      forecast vs previous so the trader has a pre-baked scenario.
#   3. Avoid mixed data  --> when two same-currency releases land in the same
#      window with conflicting outcomes (one beat, one miss), surface a
#      "MIXED -- avoid" flag so the bot/operator stands aside.
#   4. Intervention risk --> hard-coded warnings for JPY when USDJPY trades in
#      historically intervention-prone zones (155+, 160+).

_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map a trading pair to the currencies whose calendar events matter most.
_PAIR_CCYS: Dict[str, List[str]] = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "USDCAD": ["USD", "CAD"],
    "USDCHF": ["USD", "CHF"],
    "AUDUSD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "AUDJPY": ["AUD", "JPY"],
    "EURGBP": ["EUR", "GBP"],
    "XAUUSD": ["USD"],   # gold reacts mostly to USD-side data
    "XAGUSD": ["USD"],
}

# Releases where "higher number = stronger currency". For most metrics this is
# the default; exceptions (e.g. unemployment rate, jobless claims, CPI for
# bonds) are listed below so we can flip the bias correctly.
_LOWER_IS_HAWKISH = (
    "unemployment rate",
    "unemployment claims",
    "jobless claims",
    "trade balance",   # negative balance widens => bearish
)
# Higher inflation = hawkish for the currency (rate-hike pressure).
# Higher GDP, retail sales, PMI, employment = hawkish.
# We intentionally don't try to second-guess every release; the classifier
# falls back to "neutral" when it doesn't know.
_HAWKISH_HINTS = (
    "cpi", "ppi", "inflation",
    "gdp", "retail sales", "pmi", "ism",
    "employment change", "non-farm", "nfp", "payroll",
    "average hourly earnings", "wages",
    "core",
    "manufacturing production", "industrial production",
    "consumer confidence", "consumer sentiment",
    "interest rate", "rate decision", "policy rate",
)


class _CalendarCache:
    """5-minute cached fetch of the weekly calendar."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._fetched_at: float = 0.0
        self._error: Optional[str] = None

    def get(self, force: bool = False) -> Dict[str, Any]:
        now = _time.time()
        with self._lock:
            stale = (now - self._fetched_at) > self.ttl
            if not force and not stale and self._events:
                return self._snapshot()
        events, err = _fetch_calendar()
        with self._lock:
            if events:
                self._events = events
                self._fetched_at = now
                self._error = None
            else:
                self._error = err
                if not self._events:
                    self._fetched_at = now
            return self._snapshot()

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "events": list(self._events),
            "fetched_at": datetime.fromtimestamp(self._fetched_at, tz=timezone.utc).isoformat()
                          if self._fetched_at else None,
            "ttl_seconds": self.ttl,
            "error": self._error,
        }


_CALENDAR_CACHE = _CalendarCache(ttl_seconds=300)


def _parse_ff_number(raw: Optional[str]) -> Optional[float]:
    """ForexFactory expresses numbers as strings with units (e.g. '3.2%',
    '250K', '-1.5B'). We strip the formatting so we can compare them."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "--"):
        return None
    s = s.replace(",", "")
    mult = 1.0
    if s.endswith("%"):
        s = s[:-1]
    if s.endswith("K") or s.endswith("k"):
        mult, s = 1_000.0, s[:-1]
    elif s.endswith("M") or s.endswith("m"):
        mult, s = 1_000_000.0, s[:-1]
    elif s.endswith("B") or s.endswith("b"):
        mult, s = 1_000_000_000.0, s[:-1]
    elif s.endswith("T") or s.endswith("t"):
        mult, s = 1_000_000_000_000.0, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _direction_for(title: str) -> int:
    """Return +1 if higher value is hawkish for the currency, -1 if lower is."""
    lower = title.lower()
    if any(k in lower for k in _LOWER_IS_HAWKISH):
        return -1
    return 1


def _classify_outcome(title: str, forecast: Optional[float],
                      previous: Optional[float], actual: Optional[float]) -> Dict[str, Any]:
    """Compare actual to forecast (or previous, if forecast is missing) and
    label the surprise as beat / miss / inline / pending."""
    out: Dict[str, Any] = {
        "outcome": "pending",          # pending | beat | miss | inline
        "bias": "neutral",             # neutral | hawkish | dovish
        "delta_pct": None,
    }
    if actual is None:
        # Pre-release: lean on forecast vs previous as a *expectation* hint.
        if forecast is not None and previous is not None:
            direction = _direction_for(title)
            diff = forecast - previous
            if abs(diff) < 1e-9:
                out["bias"] = "neutral"
            else:
                out["bias"] = "hawkish" if (diff * direction > 0) else "dovish"
            try:
                out["delta_pct"] = (diff / abs(previous)) * 100.0 if previous else None
            except ZeroDivisionError:
                out["delta_pct"] = None
        return out

    ref = forecast if forecast is not None else previous
    if ref is None:
        out["outcome"] = "inline"
        return out

    direction = _direction_for(title)
    diff = actual - ref
    # Build a small tolerance band so a 0.01 wiggle isn't called a "beat".
    band = max(abs(ref) * 0.005, 0.05)
    if abs(diff) <= band:
        out["outcome"] = "inline"
        out["bias"] = "neutral"
    else:
        if diff > 0:
            out["outcome"] = "beat" if direction > 0 else "miss"
        else:
            out["outcome"] = "miss" if direction > 0 else "beat"
        out["bias"] = "hawkish" if out["outcome"] == "beat" else "dovish"
    try:
        out["delta_pct"] = (diff / abs(ref)) * 100.0 if ref else None
    except ZeroDivisionError:
        out["delta_pct"] = None
    return out


def _enrich_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate one ForexFactory record into our canonical event shape."""
    try:
        when = datetime.fromisoformat(raw["date"]).astimezone(timezone.utc)
    except (KeyError, ValueError, TypeError):
        return None
    title = (raw.get("title") or "").strip()
    if not title:
        return None
    impact = (raw.get("impact") or "").strip()
    if impact in ("Holiday", "Non-Economic"):
        return None
    fc = _parse_ff_number(raw.get("forecast"))
    prev = _parse_ff_number(raw.get("previous"))
    act = _parse_ff_number(raw.get("actual"))
    cls = _classify_outcome(title, fc, prev, act)
    return {
        "time": when.isoformat(),
        "ts": when.timestamp(),
        "currency": (raw.get("country") or "").strip(),
        "title": title,
        "impact": impact,
        "forecast": raw.get("forecast") or None,
        "previous": raw.get("previous") or None,
        "actual": raw.get("actual") or None,
        "forecast_num": fc,
        "previous_num": prev,
        "actual_num": act,
        "outcome": cls["outcome"],
        "bias": cls["bias"],
        "delta_pct": cls["delta_pct"],
    }


def _fetch_calendar() -> tuple[List[Dict[str, Any]], Optional[str]]:
    req = urllib.request.Request(
        _FF_CALENDAR_URL,
        headers={
            "User-Agent": "AxiomOmega/1.0 (+dashboard calendar widget)",
            "Accept": "application/json,*/*;q=0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("calendar fetch failed: %s", e)
        return [], str(e)
    try:
        raw = json.loads(data.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as e:
        return [], str(e)
    events: List[Dict[str, Any]] = []
    for r in raw or []:
        e = _enrich_event(r)
        if e is not None:
            events.append(e)
    events.sort(key=lambda e: e["ts"])
    return events, None


def _detect_mixed(events: List[Dict[str, Any]], window_minutes: int = 30) -> List[Dict[str, Any]]:
    """Mark events as `mixed=True` when another same-currency, same-window
    release fired with the opposite outcome. Implements the video's
    'avoid mixed data' rule."""
    by_ccy: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        by_ccy.setdefault(e["currency"], []).append(e)
    win = window_minutes * 60.0
    for grp in by_ccy.values():
        for i, e in enumerate(grp):
            if e["outcome"] == "pending":
                continue
            for j, other in enumerate(grp):
                if i == j or other["outcome"] == "pending":
                    continue
                if abs(other["ts"] - e["ts"]) > win:
                    continue
                if {e["outcome"], other["outcome"]} == {"beat", "miss"}:
                    e["mixed"] = True
                    e.setdefault("mixed_with", []).append({
                        "title": other["title"],
                        "outcome": other["outcome"],
                        "time": other["time"],
                    })
                    break
            else:
                e["mixed"] = e.get("mixed", False)
    return events


def _intervention_warnings(symbol: str, last_price: Optional[float]) -> List[Dict[str, Any]]:
    """Hard-coded BoJ-style intervention zones. The video flags this as the
    main reason to *skip* a setup even when the fundamental thesis is right."""
    warnings: List[Dict[str, Any]] = []
    sym = (symbol or "").upper()
    if sym.startswith("USDJPY") and last_price is not None:
        if last_price >= 160.0:
            warnings.append({
                "level": "high",
                "message": "USDJPY above 160 -- BoJ intervention historically likely. "
                           "Consider standing aside or trading smaller size.",
            })
        elif last_price >= 155.0:
            warnings.append({
                "level": "medium",
                "message": "USDJPY above 155 -- BoJ verbal intervention zone. "
                           "Watch for sudden reversals.",
            })
    if sym.endswith("JPY") and not sym.startswith("USDJPY") and last_price is not None:
        # Cross-yen pairs also see spillover from BoJ moves
        warnings.append({
            "level": "info",
            "message": f"{sym} is a JPY cross -- BoJ action on USDJPY can whip "
                       "this pair regardless of your fundamental thesis.",
        })
    return warnings


class _NewsCache:
    """Thread-safe in-memory cache for the merged news feed.

    The dashboard refreshes every few seconds, so we don't want each browser
    tick to trigger 5 outbound HTTP fetches. TTL keeps things fresh enough for
    a trading sidebar without being abusive to upstream providers.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []
        self._fetched_at: float = 0.0
        self._last_error: Optional[str] = None

    def get(self, force: bool = False) -> Dict[str, Any]:
        now = _time.time()
        with self._lock:
            stale = (now - self._fetched_at) > self.ttl
            if not force and not stale and self._items:
                return self._snapshot()
        # Fetch outside the lock so concurrent readers aren't blocked.
        items, err = _fetch_all_news()
        with self._lock:
            if items:
                self._items = items
                self._fetched_at = now
                self._last_error = None
            else:
                # Preserve the previous cache if every feed failed; surface the
                # error so the UI can show a hint instead of going blank.
                self._last_error = err
                if not self._items:
                    self._fetched_at = now  # avoid hot-looping on outage
            return self._snapshot()

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "items": list(self._items),
            "fetched_at": datetime.fromtimestamp(self._fetched_at, tz=timezone.utc).isoformat()
                          if self._fetched_at else None,
            "ttl_seconds": self.ttl,
            "error": self._last_error,
        }


_NEWS_CACHE = _NewsCache(ttl_seconds=300)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    txt = _HTML_TAG_RE.sub(" ", text)
    txt = txt.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"') \
             .replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")
    txt = _WS_RE.sub(" ", txt).strip()
    return txt


def _parse_pub_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _classify_tags(title: str, summary: str, base: Iterable[str]) -> List[str]:
    """Merge feed-level tags with keyword-derived tags so callers can filter by
    'gold' / 'forex' regardless of which feed an item came from."""
    blob = f"{title} {summary}".lower()
    tags = {t.lower() for t in base if t}
    if any(k in blob for k in _GOLD_KEYWORDS):
        tags.add("gold")
    if any(k in blob for k in _FOREX_KEYWORDS):
        tags.add("forex")
    return sorted(tags)


def _parse_rss(xml_text: str, feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Minimal RSS 2.0 / Atom parser. We avoid a feedparser dependency to keep
    the dashboard's footprint small; these feeds are well-formed enough."""
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS 2.0: channel/item; Atom: feed/entry. Handle both.
    rss_items = root.findall(".//item")
    if rss_items:
        for it in rss_items:
            title = _strip_html((it.findtext("title") or "").strip())
            link = (it.findtext("link") or "").strip()
            desc = _strip_html((it.findtext("description") or "").strip())
            pub = _parse_pub_date(it.findtext("pubDate"))
            if not title or not link:
                continue
            items.append({
                "title": title,
                "link": link,
                "summary": desc[:280],
                "published_at": pub.isoformat() if pub else None,
                "_published_ts": pub.timestamp() if pub else 0.0,
                "source": feed["source"],
                "tags": _classify_tags(title, desc, feed.get("tags", [])),
            })
        return items

    # Atom fallback
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns):
        title = _strip_html((entry.findtext("a:title", default="", namespaces=ns) or "").strip())
        link_el = entry.find("a:link", ns)
        link = link_el.get("href", "").strip() if link_el is not None else ""
        summary = _strip_html((entry.findtext("a:summary", default="", namespaces=ns) or "").strip())
        updated = entry.findtext("a:updated", default="", namespaces=ns) \
                  or entry.findtext("a:published", default="", namespaces=ns)
        pub = None
        if updated:
            try:
                pub = datetime.fromisoformat(updated.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                pub = None
        if not title or not link:
            continue
        items.append({
            "title": title,
            "link": link,
            "summary": summary[:280],
            "published_at": pub.isoformat() if pub else None,
            "_published_ts": pub.timestamp() if pub else 0.0,
            "source": feed["source"],
            "tags": _classify_tags(title, summary, feed.get("tags", [])),
        })
    return items


def _fetch_one_feed(feed: Dict[str, Any], timeout: float = 6.0) -> List[Dict[str, Any]]:
    req = urllib.request.Request(
        feed["url"],
        headers={
            # Some providers (Investing.com, Kitco) reject default Python UA.
            "User-Agent": "AxiomOmega/1.0 (+dashboard news widget)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*;q=0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("news fetch failed for %s: %s", feed["source"], e)
        return []
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []
    return _parse_rss(text, feed)


def _fetch_all_news() -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Fan-out fetch across all feeds, dedup by link, sort newest first."""
    threads: List[threading.Thread] = []
    results: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[str] = []

    def worker(f: Dict[str, Any]) -> None:
        try:
            results[f["source"]] = _fetch_one_feed(f)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f['source']}: {e}")
            results[f["source"]] = []

    for f in _NEWS_FEEDS:
        t = threading.Thread(target=worker, args=(f,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=8.0)

    merged: Dict[str, Dict[str, Any]] = {}
    for items in results.values():
        for it in items:
            key = it.get("link") or it.get("title")
            if not key:
                continue
            # First seen wins, but prefer the entry with a parsed timestamp.
            existing = merged.get(key)
            if existing is None or (existing["_published_ts"] == 0.0 and it["_published_ts"] > 0):
                merged[key] = it

    out = sorted(merged.values(), key=lambda x: x.get("_published_ts", 0.0), reverse=True)
    # Drop the internal sort key before returning
    for it in out:
        it.pop("_published_ts", None)
    err = "; ".join(errors) if errors and not out else None
    return out, err


# ---------------------------------------------------------------------- app factory
def create_app(cfg: Dict[str, Any]) -> FastAPI:
    STATE.cfg = cfg
    STATE.db_path = Path(cfg.get("journal", {}).get("db_path", "data/journal.db"))

    app = FastAPI(title="Axiom Omega Dashboard", version="0.1.0")
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --------------------------------------------------------------- routes
    @app.get("/", include_in_schema=False)
    def index():
        idx = static_dir / "index.html"
        if not idx.exists():
            raise HTTPException(500, "index.html not found")
        return FileResponse(idx)

    @app.get("/api/symbols")
    def api_symbols(seed: Optional[str] = None):
        """List of symbols the dashboard can switch between (journal + config default
        + any seed overrides). ``seed`` is a comma-separated list, e.g.
        ``?seed=GBPUSD,USDJPY,USDCAD`` to force-include pairs that haven't traded yet."""
        default = cfg["trading"]["symbol"]
        seed_list = [s.strip() for s in (seed or "").split(",") if s.strip()] if seed else None
        # Also pick up any default seed configured on the dashboard itself
        cfg_seed = cfg.get("dashboard", {}).get("seed_symbols")
        if cfg_seed and not seed_list:
            seed_list = list(cfg_seed)
        return {
            "default": default,
            "symbols": _available_symbols(default, seed=seed_list),
        }

    @app.get("/api/status")
    def api_status(symbol: Optional[str] = None):
        client = _try_mt5()
        default_symbol = cfg["trading"]["symbol"]
        sym = _resolve_symbol(symbol, default_symbol)
        timeframe = cfg["trading"]["timeframe"]
        mode = cfg["trading"].get("mode", "paper")
        out: Dict[str, Any] = {
            "symbol": sym,
            "timeframe": timeframe,
            "mode": mode,
            "magic": cfg["broker"].get("magic"),
            "mt5_connected": False,
            "balance": None,
            "equity": None,
            "currency": None,
            "server": None,
            "last_price": None,
            "now_utc": datetime.now(tz=timezone.utc).isoformat(),
        }
        if client is None:
            return out
        try:
            import MetaTrader5 as mt5  # type: ignore
            info = mt5.account_info()
            tick = mt5.symbol_info_tick(sym)
            out["mt5_connected"] = True
            if info is not None:
                out["balance"] = float(info.balance)
                out["equity"] = float(info.equity)
                out["currency"] = str(info.currency)
                out["server"] = str(info.server)
            if tick is not None:
                out["last_price"] = float((tick.ask + tick.bid) / 2.0)
                out["bid"] = float(tick.bid)
                out["ask"] = float(tick.ask)
        except Exception as e:  # noqa: BLE001
            log.warning("status: mt5 query failed: %s", e)
        return out

    @app.get("/api/candles")
    def api_candles(
        bars: int = Query(500, ge=1, le=5000),
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        client = _try_mt5()
        if client is None:
            return JSONResponse({"error": "mt5_unavailable", "candles": []}, status_code=200)
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        tf = timeframe or cfg["trading"]["timeframe"]
        try:
            df = client.get_rates(sym, tf, bars)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e), "candles": []}, status_code=200)
        candles = [
            {
                "time": int(idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for idx, row in df.iterrows()
        ]
        return {"symbol": sym, "timeframe": tf, "candles": candles}

    @app.get("/api/positions")
    def api_positions(symbol: Optional[str] = None):
        """Open positions: from MT5 in demo mode, from journal in paper mode."""
        mode = cfg["trading"].get("mode", "paper")
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        if mode == "demo":
            client = _try_mt5()
            if client is None:
                return {"mode": mode, "positions": []}
            try:
                pos = client.open_positions(sym)
            except Exception as e:  # noqa: BLE001
                return {"mode": mode, "positions": [], "error": str(e)}
            return {
                "mode": mode,
                "positions": [
                    {
                        "ticket": p.ticket,
                        "side": p.side,
                        "volume": p.volume,
                        "entry": p.price_open,
                        "sl": p.sl,
                        "tp": p.tp,
                        "pnl": p.profit,
                        "opened_at": p.time_open.isoformat(),
                    }
                    for p in pos
                ],
            }
        # paper mode: read open rows from the journal, filtered by symbol
        with _db() as c:
            rows = c.execute(
                "SELECT ticket, side, volume, entry_price AS entry, sl, tp, "
                "       opened_at, atr "
                "  FROM trades WHERE closed_at IS NULL AND symbol = ? "
                " ORDER BY opened_at DESC",
                (sym,),
            ).fetchall()
        return {"mode": mode, "positions": [_row_to_dict(r) for r in rows]}

    @app.get("/api/trades")
    def api_trades(
        limit: int = Query(50, ge=1, le=500),
        symbol: Optional[str] = None,
    ):
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        with _db() as c:
            rows = c.execute(
                "SELECT ticket, symbol, side, volume, entry_price, sl, tp, atr, "
                "       risk_money, risk_pct, reason, opened_at, closed_at, "
                "       close_price, close_reason, pnl, outcome "
                "  FROM trades WHERE closed_at IS NOT NULL AND symbol = ? "
                " ORDER BY closed_at DESC LIMIT ?",
                (sym, int(limit)),
            ).fetchall()
        return {"trades": [_row_to_dict(r) for r in rows]}

    @app.get("/api/signals")
    def api_signals(
        limit: int = Query(50, ge=1, le=500),
        symbol: Optional[str] = None,
    ):
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        with _db() as c:
            rows = c.execute(
                "SELECT id, bar_time, symbol, side, reason, entry_ref, atr, "
                "       proba, acted, skip_reason, created_at "
                "  FROM signals WHERE symbol = ? "
                " ORDER BY id DESC LIMIT ?",
                (sym, int(limit)),
            ).fetchall()
        return {"signals": [_row_to_dict(r) for r in rows]}

    @app.get("/api/equity_curve")
    def api_equity_curve(symbol: Optional[str] = None):
        """Cumulative pnl over closed trades (UTC). If symbol is given, scope to that pair;
        otherwise show portfolio-wide cumulative pnl across all symbols."""
        sym = _resolve_symbol(symbol, "") or None
        with _db() as c:
            if sym:
                rows = c.execute(
                    "SELECT closed_at, pnl FROM trades "
                    " WHERE closed_at IS NOT NULL AND symbol = ? "
                    " ORDER BY closed_at",
                    (sym,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT closed_at, pnl FROM trades "
                    " WHERE closed_at IS NOT NULL "
                    " ORDER BY closed_at"
                ).fetchall()
        cum = 0.0
        points = []
        for r in rows:
            cum += float(r["pnl"] or 0.0)
            points.append({"time": r["closed_at"], "cum_pnl": cum})
        return {"symbol": sym, "points": points}

    @app.get("/api/stats")
    def api_stats(symbol: Optional[str] = None):
        """Per-symbol stats when ?symbol= is given, otherwise portfolio-wide."""
        sym = _resolve_symbol(symbol, "") or None
        where_sym = " AND symbol = ?" if sym else ""
        params = (sym,) if sym else ()
        with _db() as c:
            row = c.execute(
                "SELECT COUNT(*)               AS n,"
                "       SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) AS wins,"
                "       SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END) AS losses,"
                "       COALESCE(SUM(pnl), 0)  AS total_pnl,"
                "       COALESCE(AVG(pnl), 0)  AS avg_pnl"
                "  FROM trades WHERE outcome IS NOT NULL" + where_sym,
                params,
            ).fetchone()
            today_iso = (datetime.now(tz=timezone.utc) - timedelta(hours=24)).isoformat()
            today = c.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(pnl),0) AS pnl "
                "  FROM trades WHERE closed_at IS NOT NULL AND closed_at > ?" + where_sym,
                (today_iso, *params),
            ).fetchone()
            sig = c.execute(
                "SELECT COUNT(*) AS n,"
                "       SUM(CASE WHEN acted=1 THEN 1 ELSE 0 END) AS acted "
                "  FROM signals" + (" WHERE symbol = ?" if sym else ""),
                params,
            ).fetchone()
            last_loss = c.execute(
                "SELECT closed_at FROM trades WHERE outcome=0" + where_sym +
                " ORDER BY closed_at DESC LIMIT 1",
                params,
            ).fetchone()
        n = int(row["n"] or 0)
        wins = int(row["wins"] or 0)
        return {
            "symbol": sym,
            "closed_trades": n,
            "wins": wins,
            "losses": int(row["losses"] or 0),
            "win_rate": (wins / n) if n else 0.0,
            "total_pnl": float(row["total_pnl"] or 0.0),
            "avg_pnl": float(row["avg_pnl"] or 0.0),
            "trades_24h": int(today["n"] or 0),
            "pnl_24h": float(today["pnl"] or 0.0),
            "signals_total": int(sig["n"] or 0),
            "signals_acted": int(sig["acted"] or 0),
            "last_loss_at": last_loss["closed_at"] if last_loss else None,
        }

    @app.get("/api/chart_markers")
    def api_chart_markers(
        limit: int = Query(100, ge=1, le=500),
        symbol: Optional[str] = None,
    ):
        """Trade entries/exits as chart markers (most recent first), filtered by symbol."""
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        with _db() as c:
            rows = c.execute(
                "SELECT ticket, side, entry_price, opened_at, "
                "       close_price, closed_at, close_reason, outcome, reason "
                "  FROM trades WHERE symbol = ? ORDER BY opened_at DESC LIMIT ?",
                (sym, int(limit)),
            ).fetchall()
        markers: List[Dict[str, Any]] = []
        for r in rows:
            try:
                t_in = int(datetime.fromisoformat(r["opened_at"]).timestamp())
            except Exception:  # noqa: BLE001
                continue
            markers.append({
                "time": t_in,
                "position": "belowBar" if r["side"] == "buy" else "aboveBar",
                "color": "#2F6F4E" if r["side"] == "buy" else "#B8423A",
                "shape": "arrowUp" if r["side"] == "buy" else "arrowDown",
                "text": f"{r['side']} #{r['ticket']}",
            })
            if r["closed_at"]:
                try:
                    t_out = int(datetime.fromisoformat(r["closed_at"]).timestamp())
                except Exception:  # noqa: BLE001
                    continue
                outcome_color = "#2F6F4E" if r["outcome"] == 1 else "#B8423A"
                markers.append({
                    "time": t_out,
                    "position": "aboveBar" if r["side"] == "buy" else "belowBar",
                    "color": outcome_color,
                    "shape": "circle",
                    "text": f"close {r['close_reason'] or ''}".strip(),
                })
        # Sort ascending by time for chart consumption
        markers.sort(key=lambda m: m["time"])
        return {"markers": markers}

    @app.get("/api/news")
    def api_news(
        limit: int = Query(20, ge=1, le=100),
        tag: Optional[str] = Query(None, description="Filter: 'gold', 'forex', or comma-separated"),
        refresh: bool = Query(False, description="Force-refresh the cache"),
    ):
        """Aggregated forex / gold news from public RSS feeds. Cached server-side
        for 5 minutes (configurable via ``_NewsCache.ttl``)."""
        snap = _NEWS_CACHE.get(force=refresh)
        items = snap["items"]
        if tag:
            wanted = {t.strip().lower() for t in tag.split(",") if t.strip()}
            if wanted:
                items = [it for it in items if wanted.intersection(set(it.get("tags", [])))]
        return {
            "items": items[: int(limit)],
            "total": len(items),
            "fetched_at": snap["fetched_at"],
            "ttl_seconds": snap["ttl_seconds"],
            "error": snap["error"],
        }

    @app.get("/api/calendar")
    def api_calendar(
        symbol: Optional[str] = None,
        impact: str = Query("High,Medium", description="Comma-separated impact filter"),
        upcoming_hours: int = Query(72, ge=1, le=336),
        recent_hours: int = Query(6, ge=0, le=48),
        refresh: bool = Query(False),
    ):
        """Economic calendar relevant to the active trading pair.

        Drives the WelcomeHomeTrading-style 'trade-the-news' workflow:
        upcoming releases for if-then planning, recent releases with
        beat/miss/inline outcomes, mixed-data warnings, and a per-currency
        bias summary."""
        snap = _CALENDAR_CACHE.get(force=refresh)
        all_events = snap["events"]
        # Window: keep recent results visible briefly so the trader can read
        # the surprise that just printed.
        now_ts = _time.time()
        win_low = now_ts - recent_hours * 3600
        win_high = now_ts + upcoming_hours * 3600
        events = [e for e in all_events if win_low <= e["ts"] <= win_high]
        # Impact filter
        wanted_impacts = {i.strip() for i in impact.split(",") if i.strip()}
        if wanted_impacts:
            events = [e for e in events if e["impact"] in wanted_impacts]
        # Pair relevance
        sym = _resolve_symbol(symbol, cfg["trading"]["symbol"])
        ccys = _PAIR_CCYS.get(sym.upper(), [])
        if ccys:
            events = [e for e in events if e["currency"] in ccys]
        # Tag mixed-data conflicts (uses *all* same-currency results, not just
        # the filtered slice, so a mixed pair across impact tiers still flags)
        _detect_mixed(events)

        # Per-currency net bias from completed surprises
        bias_by_ccy: Dict[str, Dict[str, int]] = {}
        for e in events:
            if e["outcome"] in ("beat", "miss"):
                b = bias_by_ccy.setdefault(e["currency"], {"hawkish": 0, "dovish": 0, "mixed": 0})
                if e.get("mixed"):
                    b["mixed"] += 1
                elif e["bias"] == "hawkish":
                    b["hawkish"] += 1
                elif e["bias"] == "dovish":
                    b["dovish"] += 1

        # Live price for intervention warning
        last_price = None
        try:
            client = _try_mt5()
            if client is not None:
                import MetaTrader5 as mt5  # type: ignore
                tick = mt5.symbol_info_tick(sym)
                if tick is not None:
                    last_price = float((tick.ask + tick.bid) / 2.0)
        except Exception:  # noqa: BLE001
            last_price = None

        warnings = _intervention_warnings(sym, last_price)

        # Find the next high-impact release for the active pair (drives the
        # 'plan if-then before the release' panel).
        next_high = None
        for e in events:
            if e["ts"] >= now_ts and e["impact"] == "High":
                next_high = e
                break

        return {
            "symbol": sym,
            "currencies": ccys,
            "events": events,
            "next_high": next_high,
            "bias": bias_by_ccy,
            "warnings": warnings,
            "last_price": last_price,
            "fetched_at": snap["fetched_at"],
            "ttl_seconds": snap["ttl_seconds"],
            "now_utc": datetime.now(tz=timezone.utc).isoformat(),
            "error": snap["error"],
        }

    return app


# ---------------------------------------------------------------------- entrypoint
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Axiom Omega dashboard")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    app = create_app(cfg)

    import uvicorn
    log.info("Dashboard starting on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
