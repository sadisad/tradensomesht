/* Axiom Omega dashboard - vanilla JS */
(function () {
  "use strict";

  const REFRESH_MS = 5000;
  const CHART_REFRESH_MS = 15000;
  const NEWS_REFRESH_MS = 120000;
  const CALENDAR_REFRESH_MS = 60000;
  const COUNTDOWN_MS = 1000;

  let chart = null;
  let candleSeries = null;
  let equityChart = null;
  let equitySeries = null;
  let lastCandleTime = 0;
  let lastChartLoad = 0;
  let lastNewsLoad = 0;
  let lastCalendarLoad = 0;
  let calendarImpact = "High,Medium";
  let lastCalendarData = null;
  let newsTag = "";

  // Currently-selected symbol. Initialised from localStorage on boot, then
  // overwritten once /api/symbols replies with the canonical list.
  let currentSymbol = (typeof localStorage !== "undefined"
    ? localStorage.getItem("rt:symbol")
    : null) || "";
  let knownSymbols = [];

  // Currently-selected timeframe. ``""`` = use the bot's configured default
  // (whatever ``cfg.trading.timeframe`` resolves to on the backend). When the
  // user picks a TF in the topbar we override it for chart queries only --
  // the bot itself keeps trading its configured TF.
  let currentTimeframe = (typeof localStorage !== "undefined"
    ? localStorage.getItem("rt:tf")
    : null) || "";

  // ---- Theme: pull live values from CSS so charts match the rest of the UI.
  // Reading them once at boot is fine because the theme is static per page load.
  const css = getComputedStyle(document.documentElement);
  const theme = {
    bg:      (css.getPropertyValue("--bg-chart")      || "#FBF7EE").trim(),
    text:    (css.getPropertyValue("--ink")           || "#1B1A17").trim(),
    grid:    (css.getPropertyValue("--hairline-soft") || "rgba(226,217,196,0.55)").trim(),
    border:  (css.getPropertyValue("--hairline")      || "#E2D9C4").trim(),
    win:     (css.getPropertyValue("--win")           || "#2F6F4E").trim(),
    loss:    (css.getPropertyValue("--loss")          || "#B8423A").trim(),
    accent:  (css.getPropertyValue("--accent")        || "#C84E2C").trim(),
  };

  // ----------------------------------------------------------------- helpers
  function fmtNum(v, digits) {
    if (v === null || v === undefined || isNaN(v)) return "--";
    return Number(v).toLocaleString(undefined, {
      minimumFractionDigits: digits, maximumFractionDigits: digits,
    });
  }
  function fmtMoney(v, ccy) {
    if (v === null || v === undefined || isNaN(v)) return "--";
    const sign = v >= 0 ? "" : "-";
    return `${sign}${ccy || ""}${fmtNum(Math.abs(v), 2)}`;
  }
  function pnlClass(v) { return v > 0 ? "pnl-pos" : v < 0 ? "pnl-neg" : ""; }
  function sideClass(s) { return s === "buy" ? "side-buy" : "side-sell"; }
  function fmtTime(iso) {
    if (!iso) return "--";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toISOString().replace("T", " ").slice(0, 19) + "Z";
  }
  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  }
  function setText(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }

  /** Append the currently-selected symbol as a query param to ``url``. */
  function withSymbol(url) {
    if (!currentSymbol) return url;
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}symbol=${encodeURIComponent(currentSymbol)}`;
  }

  /** Append the currently-selected timeframe override (if any). The chart
   *  endpoint already falls back to the bot's configured TF when this param
   *  is omitted, so empty currentTimeframe means 'use the default'. */
  function withTimeframe(url) {
    if (!currentTimeframe) return url;
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}timeframe=${encodeURIComponent(currentTimeframe)}`;
  }

  // ----------------------------------------------------------------- chart
  function ensureChart() {
    if (chart) return;
    const el = document.getElementById("chart");
    chart = LightweightCharts.createChart(el, {
      layout: { background: { color: theme.bg }, textColor: theme.text },
      grid: { vertLines: { color: theme.grid }, horzLines: { color: theme.grid } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: theme.border },
      rightPriceScale: { borderColor: theme.border },
      crosshair: { mode: 1 },
      autoSize: true,
    });
    candleSeries = chart.addCandlestickSeries({
      upColor: theme.win, downColor: theme.loss,
      borderUpColor: theme.win, borderDownColor: theme.loss,
      wickUpColor: theme.win, wickDownColor: theme.loss,
    });
    window.addEventListener("resize", () => chart && chart.applyOptions({}));
  }

  async function loadCandlesFull() {
    ensureChart();
    try {
      const data = await getJSON(withTimeframe(withSymbol("/api/candles?bars=500")));
      if (data.error) {
        setText("chart-sub", `chart unavailable: ${data.error}`);
        return;
      }
      candleSeries.setData(data.candles);
      lastCandleTime = data.candles.length ? data.candles[data.candles.length - 1].time : 0;
      setText("chart-title", `${data.symbol} (${data.timeframe})`);
      setText("chart-sub", `${data.candles.length} bars`);
      // Markers
      const m = await getJSON(withSymbol("/api/chart_markers?limit=200"));
      if (m.markers) candleSeries.setMarkers(m.markers);
      lastChartLoad = Date.now();
    } catch (e) {
      setText("chart-sub", `chart error: ${e.message}`);
    }
  }

  async function refreshLastCandle() {
    if (!candleSeries) return;
    try {
      const data = await getJSON(withTimeframe(withSymbol("/api/candles?bars=2")));
      if (!data.candles || !data.candles.length) return;
      data.candles.forEach((c) => {
        if (c.time >= lastCandleTime) {
          candleSeries.update(c);
          lastCandleTime = c.time;
        }
      });
    } catch (_) { /* ignore transient errors */ }
  }

  // ----------------------------------------------------------------- equity curve
  function ensureEquityChart() {
    if (equityChart) return;
    const el = document.getElementById("equity-chart");
    equityChart = LightweightCharts.createChart(el, {
      layout: { background: { color: theme.bg }, textColor: theme.text },
      grid: { vertLines: { color: theme.grid }, horzLines: { color: theme.grid } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: theme.border },
      rightPriceScale: { borderColor: theme.border },
      crosshair: { mode: 1 },
      autoSize: true,
    });
    equitySeries = equityChart.addAreaSeries({
      lineColor: theme.accent,
      topColor: "rgba(200, 78, 44, 0.22)",
      bottomColor: "rgba(200, 78, 44, 0.02)",
      lineWidth: 2,
    });
  }

  async function refreshEquityCurve() {
    ensureEquityChart();
    try {
      const r = await getJSON(withSymbol("/api/equity_curve"));
      const points = (r.points || [])
        .map((p) => {
          const d = new Date(p.time);
          if (isNaN(d.getTime())) return null;
          return { time: Math.floor(d.getTime() / 1000), value: Number(p.cum_pnl) };
        })
        .filter(Boolean);
      // lightweight-charts requires unique, ascending timestamps. Dedup by keeping the last.
      const dedup = [];
      const seen = new Map();
      points.forEach((p) => seen.set(p.time, p.value));
      Array.from(seen.entries())
        .sort((a, b) => a[0] - b[0])
        .forEach(([t, v]) => dedup.push({ time: t, value: v }));
      equitySeries.setData(dedup);
      const last = dedup.length ? dedup[dedup.length - 1].value : 0;
      const first = dedup.length ? dedup[0].value : 0;
      setText("equity-sub",
        dedup.length
          ? `${dedup.length} closed trades | last: ${fmtMoney(last, "")} | range: ${fmtMoney(first, "")} -> ${fmtMoney(last, "")}`
          : "no closed trades yet");
    } catch (e) {
      setText("equity-sub", `equity error: ${e.message}`);
    }
  }

  // ----------------------------------------------------------------- top bar / kpis
  async function refreshStatus() {
    try {
      const s = await getJSON(withSymbol("/api/status"));
      const dot = document.getElementById("conn-dot");
      if (s.mt5_connected) {
        dot.className = "dot ok";
        setText("status-line", `${s.symbol} ${s.timeframe} | ${s.server || ""}`);
      } else {
        dot.className = "dot warn";
        setText("status-line", `${s.symbol} ${s.timeframe} | MT5 offline (journal-only)`);
      }
      const ccy = s.currency ? s.currency + " " : "";
      setText("kpi-equity",  s.equity  != null ? `${ccy}${fmtNum(s.equity, 2)}`  : "--");
      setText("kpi-balance", s.balance != null ? `${ccy}${fmtNum(s.balance, 2)}` : "--");
      setText("kpi-price",   s.last_price != null ? fmtNum(s.last_price, 2) : "--");
      setText("kpi-mode",    (s.mode || "--").toUpperCase());
    } catch (e) {
      const dot = document.getElementById("conn-dot");
      if (dot) dot.className = "dot error";
      setText("status-line", `dashboard error: ${e.message}`);
    }
  }

  async function refreshStats() {
    try {
      const s = await getJSON(withSymbol("/api/stats"));
      setText("s-closed", String(s.closed_trades));
      setText("s-wr", s.closed_trades ? (s.win_rate * 100).toFixed(1) + "%" : "--");
      const pnlEl = document.getElementById("s-pnl");
      pnlEl.textContent = fmtMoney(s.total_pnl, "");
      pnlEl.className = pnlClass(s.total_pnl);
      const pnl24 = document.getElementById("s-pnl24");
      pnl24.textContent = fmtMoney(s.pnl_24h, "");
      pnl24.className = pnlClass(s.pnl_24h);
      setText("s-sig", `${s.signals_acted}/${s.signals_total}`);
      setText("s-lastloss", s.last_loss_at ? fmtTime(s.last_loss_at) : "--");
    } catch (_) {}
  }

  // ----------------------------------------------------------------- tables
  async function refreshPositions() {
    try {
      const r = await getJSON(withSymbol("/api/positions"));
      const tbody = document.querySelector("#positions-table tbody");
      const empty = document.getElementById("positions-empty");
      tbody.innerHTML = "";
      if (!r.positions || r.positions.length === 0) {
        empty.style.display = "block";
        return;
      }
      empty.style.display = "none";
      r.positions.forEach((p) => {
        const pnl = p.pnl != null ? p.pnl : null;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${p.ticket}</td>
          <td class="${sideClass(p.side)}">${p.side.toUpperCase()}</td>
          <td>${fmtNum(p.volume, 2)}</td>
          <td>${fmtNum(p.entry, 2)}</td>
          <td>${fmtNum(p.sl, 2)}</td>
          <td>${fmtNum(p.tp, 2)}</td>
          <td class="${pnl != null ? pnlClass(pnl) : ""}">${pnl != null ? fmtMoney(pnl, "") : "--"}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (_) {}
  }

  async function refreshSignals() {
    try {
      const r = await getJSON(withSymbol("/api/signals?limit=20"));
      const tbody = document.querySelector("#signals-table tbody");
      tbody.innerHTML = "";
      r.signals.forEach((s) => {
        const status = s.acted
          ? `<span class="pill acted">acted</span>`
          : `<span class="pill skipped" title="${s.skip_reason || ""}">skipped</span>`;
        const proba = s.proba != null ? (s.proba * 100).toFixed(1) + "%" : "--";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${fmtTime(s.bar_time)}</td>
          <td class="${sideClass(s.side)}">${s.side.toUpperCase()}</td>
          <td>${s.reason || ""}</td>
          <td>${proba}</td>
          <td>${status}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (_) {}
  }

  async function refreshTrades() {
    try {
      const r = await getJSON(withSymbol("/api/trades?limit=30"));
      const tbody = document.querySelector("#trades-table tbody");
      tbody.innerHTML = "";
      r.trades.forEach((t) => {
        const wl = t.outcome === 1
          ? `<span class="pill win">WIN</span>`
          : `<span class="pill loss">LOSS</span>`;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${fmtTime(t.closed_at)}</td>
          <td class="${sideClass(t.side)}">${t.side.toUpperCase()}</td>
          <td>${fmtNum(t.volume, 2)}</td>
          <td>${fmtNum(t.entry_price, 2)}</td>
          <td>${fmtNum(t.close_price, 2)}</td>
          <td>${t.close_reason || ""}</td>
          <td class="${pnlClass(t.pnl)}">${fmtMoney(t.pnl, "")}</td>
          <td>${wl}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (_) {}
  }

  // ----------------------------------------------------------------- news
  function fmtAge(iso) {
    if (!iso) return "";
    const t = Date.parse(iso);
    if (isNaN(t)) return "";
    const diff = Math.max(0, Date.now() - t);
    const m = Math.floor(diff / 60000);
    if (m < 1)  return "just now";
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.floor(h / 24);
    return `${d}d ago`;
  }

  function escapeHTML(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function renderNews(items) {
    const list = document.getElementById("news-list");
    const empty = document.getElementById("news-empty");
    if (!list) return;
    list.innerHTML = "";
    if (!items || !items.length) {
      empty.hidden = false;
      return;
    }
    empty.hidden = true;
    items.forEach((it) => {
      const tags = (it.tags || [])
        .filter((t) => t === "gold" || t === "forex")
        .map((t) => `<span class="news-tag tag-${t}">${t}</span>`)
        .join("");
      const li = document.createElement("li");
      li.className = "news-item";
      li.innerHTML = `
        <a class="news-link" href="${escapeHTML(it.link)}" target="_blank" rel="noopener noreferrer">
          <div class="news-meta">
            <span class="news-source">${escapeHTML(it.source || "")}</span>
            ${tags}
            <span class="news-age">${escapeHTML(fmtAge(it.published_at))}</span>
          </div>
          <h4 class="news-title">${escapeHTML(it.title)}</h4>
          ${it.summary ? `<p class="news-summary">${escapeHTML(it.summary)}</p>` : ""}
        </a>
      `;
      list.appendChild(li);
    });
  }

  async function refreshNews(force) {
    const sub = document.getElementById("news-sub");
    try {
      const qs = new URLSearchParams();
      qs.set("limit", "12");
      if (newsTag) qs.set("tag", newsTag);
      if (force) qs.set("refresh", "1");
      const r = await getJSON(`/api/news?${qs.toString()}`);
      renderNews(r.items);
      const fetched = r.fetched_at ? fmtAge(r.fetched_at) : "now";
      const label = newsTag ? newsTag.toUpperCase() : "ALL";
      if (sub) {
        sub.textContent = r.error
          ? `${label} | partial data | updated ${fetched}`
          : `${label} | ${r.items.length} of ${r.total} | updated ${fetched}`;
      }
      lastNewsLoad = Date.now();
    } catch (e) {
      if (sub) sub.textContent = `news error: ${e.message}`;
    }
  }

  function initNewsControls() {
    const seg = document.querySelectorAll(".seg-btn[data-news-tag]");
    if (!seg.length) return;
    seg.forEach((btn) => {
      btn.addEventListener("click", () => {
        const tag = btn.dataset.newsTag || "";
        if (tag === newsTag) return;
        newsTag = tag;
        seg.forEach((b) => b.classList.toggle("is-active", b === btn));
        refreshNews(false);
      });
    });
  }

  // ----------------------------------------------------------------- calendar
  // Implements the WelcomeHomeTrading 'trade-the-news' workflow:
  //  - Plan if-then BEFORE release: countdown + scenarios for the next high event
  //  - Fundamental = WHY: per-currency bias from completed surprises this week
  //  - Avoid mixed data: explicit MIXED pill and warning
  //  - Intervention risk: server-side warnings rendered above the calendar list

  function fmtCountdown(ms) {
    if (ms == null || isNaN(ms)) return "--";
    const past = ms < 0;
    const abs = Math.abs(ms);
    const totalSec = Math.floor(abs / 1000);
    const d = Math.floor(totalSec / 86400);
    const h = Math.floor((totalSec % 86400) / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    let out;
    if (d > 0) out = `${d}d ${h}h ${String(m).padStart(2, "0")}m`;
    else if (h > 0) out = `${h}h ${String(m).padStart(2, "0")}m`;
    else out = `${m}m ${String(s).padStart(2, "0")}s`;
    return past ? `${out} ago` : `in ${out}`;
  }

  function fmtClock(iso) {
    if (!iso) return "--";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "--";
    const day = d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mm = String(d.getUTCMinutes()).padStart(2, "0");
    return `${day} ${hh}:${mm}Z`;
  }

  function impactBadge(impact) {
    const cls = (impact || "").toLowerCase();
    return `<span class="impact impact-${cls}">${impact || "--"}</span>`;
  }

  function outcomePill(e) {
    if (e.mixed) return `<span class="pill mixed" title="Mixed data on the same currency">MIXED</span>`;
    if (e.priced_in && e.outcome === "pending") {
      return `<span class="pill priced-in" title="Rate change matches previous -- statement drives price">PRICED IN</span>`;
    }
    if (e.statement_tone === "hawkish") return `<span class="pill beat" title="Hawkish statement scanned from news">HAWKISH STMT</span>`;
    if (e.statement_tone === "dovish")  return `<span class="pill miss" title="Dovish statement scanned from news">DOVISH STMT</span>`;
    if (e.statement_tone === "mixed")   return `<span class="pill mixed" title="Statement keywords mixed">MIXED STMT</span>`;
    if (e.outcome === "beat")   return `<span class="pill beat">BEAT</span>`;
    if (e.outcome === "miss")   return `<span class="pill miss">MISS</span>`;
    if (e.outcome === "inline") return `<span class="pill inline">INLINE</span>`;
    if (e.bias === "hawkish")   return `<span class="pill expect-hawk">expect hawkish</span>`;
    if (e.bias === "dovish")    return `<span class="pill expect-dove">expect dovish</span>`;
    return `<span class="pill pending">pending</span>`;
  }

  function renderPlan(data) {
    const body = document.getElementById("plan-body");
    const sub = document.getElementById("plan-sub");
    if (!body) return;
    const e = data && data.next_high;
    if (!e) {
      body.innerHTML = `<p class="empty">No high-impact release in the next 72h&hellip;</p>`;
      if (sub) sub.textContent = data && data.symbol ? `for ${data.symbol}` : "--";
      return;
    }
    const ms = (e.ts * 1000) - Date.now();
    const ccy = e.currency;
    const fc = e.forecast || "--";
    const prev = e.previous || "--";

    // Branch: priced-in rate decision -> the *statement* is the trade, not the
    // headline number. Per the WelcomeHomeTrading video on RBA: forecast ==
    // previous means the move comes from tone, not the rate itself.
    if (e.is_rate_decision && e.priced_in) {
      body.innerHTML = `
        <div class="plan-headline">
          <span class="plan-ccy">${ccy}</span>
          <span class="plan-title">${escapeHTML(e.title)}</span>
          ${impactBadge(e.impact)}
          <span class="pill priced-in">PRICED IN</span>
        </div>
        <div class="plan-meta">
          <span class="plan-when" data-plan-ts="${e.ts}">${fmtCountdown(ms)}</span>
          <span class="muted">${fmtClock(e.time)}</span>
          <span class="muted">forecast ${escapeHTML(fc)} == previous ${escapeHTML(prev)}</span>
        </div>
        <p class="plan-lean">
          Rate decision is <strong>already priced in</strong>. The statement
          tone &mdash; not the number &mdash; will move price.
        </p>
        <div class="plan-scenarios">
          <div class="plan-scen scen-up">
            <span class="scen-tag">If statement is hawkish</span>
            <p>${ccy} likely <strong>strengthens</strong>. Wait for break-of-structure
               on the chart, then enter pro-${ccy}. Take 90% off at first
               liquidity / TP1, trail the rest at breakeven.</p>
          </div>
          <div class="plan-scen scen-down">
            <span class="scen-tag">If statement is dovish</span>
            <p>${ccy} likely <strong>weakens</strong>. Same playbook, opposite
               direction. Use the chop after the headline to identify the
               break.</p>
          </div>
          <div class="plan-scen scen-mixed">
            <span class="scen-tag">If tone is balanced</span>
            <p><strong>Stand aside.</strong> No fundamental edge means no
               trade &mdash; protect capital for the next setup.</p>
          </div>
        </div>
      `;
      if (sub) sub.textContent = `for ${data.symbol} | ${ccy} | rate decision`;
      return;
    }

    // Default branch: surprise-driven event (CPI, NFP, GDP, etc).
    const direction = _direction(e.title);
    const lean = e.bias === "hawkish"
      ? `Forecast leans <span class="hawk">hawkish</span> for ${ccy}`
      : e.bias === "dovish"
        ? `Forecast leans <span class="dove">dovish</span> for ${ccy}`
        : `Forecast roughly in line with previous`;
    const arrowUp = direction > 0 ? "above forecast" : "below forecast";
    const arrowDown = direction > 0 ? "below forecast" : "above forecast";
    body.innerHTML = `
      <div class="plan-headline">
        <span class="plan-ccy">${ccy}</span>
        <span class="plan-title">${escapeHTML(e.title)}</span>
        ${impactBadge(e.impact)}
      </div>
      <div class="plan-meta">
        <span class="plan-when" data-plan-ts="${e.ts}">${fmtCountdown(ms)}</span>
        <span class="muted">${fmtClock(e.time)}</span>
        <span class="muted">forecast ${escapeHTML(fc)} | previous ${escapeHTML(prev)}</span>
      </div>
      <p class="plan-lean">${lean}.</p>
      <div class="plan-scenarios">
        <div class="plan-scen scen-up">
          <span class="scen-tag">If actual ${arrowUp}</span>
          <p>${ccy} likely <strong>strengthens</strong>. Look for break-of-structure
             continuation in the direction that favors ${ccy}.</p>
        </div>
        <div class="plan-scen scen-down">
          <span class="scen-tag">If actual ${arrowDown}</span>
          <p>${ccy} likely <strong>weakens</strong>. Look for break-of-structure in
             the direction that fades ${ccy}.</p>
        </div>
        <div class="plan-scen scen-mixed">
          <span class="scen-tag">If mixed / inline</span>
          <p><strong>Stand aside.</strong> Reaction will be choppy and your
             fundamental edge disappears.</p>
        </div>
      </div>
    `;
    if (sub) sub.textContent = `for ${data.symbol} | ${ccy}`;
  }

  function _direction(title) {
    // Mirrors backend _direction_for: lower-is-hawkish for these titles.
    const t = String(title || "").toLowerCase();
    const inverted = ["unemployment rate", "unemployment claims",
                      "jobless claims", "trade balance"];
    return inverted.some((k) => t.includes(k)) ? -1 : 1;
  }

  function renderBias(data) {
    const body = document.getElementById("bias-body");
    if (!body) return;
    const bias = (data && data.bias) || {};
    const ccys = (data && data.currencies) || Object.keys(bias);
    const rows = ccys
      .filter((c) => bias[c])
      .map((c) => {
        const b = bias[c];
        const net = (b.hawkish || 0) - (b.dovish || 0);
        const cls = net > 0 ? "hawk" : net < 0 ? "dove" : "neutral";
        const label = net > 0 ? "Hawkish" : net < 0 ? "Dovish" : "Neutral";
        const mixed = b.mixed ? `<span class="bias-mixed">${b.mixed} mixed</span>` : "";
        return `
          <li class="bias-row">
            <span class="bias-ccy">${c}</span>
            <span class="bias-tag ${cls}">${label}</span>
            <span class="muted">${b.hawkish || 0} beats / ${b.dovish || 0} misses</span>
            ${mixed}
          </li>`;
      });
    if (!rows.length) {
      body.innerHTML = `<p class="empty">No surprises printed yet for this pair&hellip;</p>`;
      return;
    }
    body.innerHTML = `<ul class="bias-list">${rows.join("")}</ul>`;
  }

  function renderCalendar(data) {
    const list = document.getElementById("calendar-list");
    const empty = document.getElementById("calendar-empty");
    const sub = document.getElementById("calendar-sub");
    const pair = document.getElementById("calendar-pair");
    const warnBox = document.getElementById("calendar-warnings");
    if (!list) return;
    if (pair) pair.textContent = data.symbol || "--";

    // Warnings (intervention zones, etc.)
    if (warnBox) {
      const ws = data.warnings || [];
      if (ws.length) {
        warnBox.hidden = false;
        warnBox.innerHTML = ws.map((w) =>
          `<div class="warn warn-${w.level}">${escapeHTML(w.message)}</div>`
        ).join("");
      } else {
        warnBox.hidden = true;
        warnBox.innerHTML = "";
      }
    }

    list.innerHTML = "";
    const events = data.events || [];
    if (!events.length) {
      empty.hidden = false;
      if (sub) sub.textContent = data.error ? `error: ${data.error}` : "";
      return;
    }
    empty.hidden = true;
    const now = Date.now();
    events.forEach((e) => {
      const ms = (e.ts * 1000) - now;
      const past = ms < 0;
      const li = document.createElement("li");
      li.className = `cal-event ${past ? "is-past" : "is-upcoming"} ${e.mixed ? "is-mixed" : ""}`;
      li.innerHTML = `
        <div class="cal-time">
          <span class="cal-clock">${fmtClock(e.time)}</span>
          <span class="cal-countdown" data-plan-ts="${e.ts}">${fmtCountdown(ms)}</span>
        </div>
        <div class="cal-meta">
          <span class="cal-ccy">${e.currency}</span>
          ${impactBadge(e.impact)}
        </div>
        <div class="cal-title">${escapeHTML(e.title)}</div>
        <div class="cal-numbers">
          <span class="muted">fc</span> <strong>${escapeHTML(e.forecast || "--")}</strong>
          <span class="muted">prev</span> <strong>${escapeHTML(e.previous || "--")}</strong>
          <span class="muted">act</span> <strong>${escapeHTML(e.actual || "--")}</strong>
        </div>
        <div class="cal-outcome">${outcomePill(e)}</div>
      `;
      list.appendChild(li);
    });
    if (sub) {
      const fetched = data.fetched_at ? fmtAge(data.fetched_at) : "--";
      sub.textContent = data.error
        ? `partial data | updated ${fetched}`
        : `${events.length} events | updated ${fetched}`;
    }
  }

  function renderNewsBanner(data) {
    const banner = document.getElementById("news-banner");
    const title = document.getElementById("news-banner-title");
    const cd = document.getElementById("news-banner-countdown");
    if (!banner) return;
    const e = data && data.next_high;
    if (!e) { banner.hidden = true; return; }
    const ms = (e.ts * 1000) - Date.now();
    // Fire window: 30 minutes before -> 5 minutes after the release.
    if (ms > 30 * 60 * 1000 || ms < -5 * 60 * 1000) {
      banner.hidden = true;
      return;
    }
    banner.hidden = false;
    if (title) title.textContent = `${e.currency} - ${e.title}`;
    if (cd) cd.textContent = fmtCountdown(ms);
    banner.classList.toggle("is-imminent", ms < 5 * 60 * 1000 && ms > -2 * 60 * 1000);
  }

  function tickCountdowns() {
    const now = Date.now();
    document.querySelectorAll("[data-plan-ts]").forEach((el) => {
      const ts = Number(el.dataset.planTs);
      if (!ts) return;
      el.textContent = fmtCountdown(ts * 1000 - now);
    });
    if (lastCalendarData) renderNewsBanner(lastCalendarData);
  }

  async function refreshCalendar(force) {
    try {
      const qs = new URLSearchParams();
      qs.set("impact", calendarImpact);
      qs.set("upcoming_hours", "168");
      qs.set("recent_hours", "12");
      if (currentSymbol) qs.set("symbol", currentSymbol);
      if (force) qs.set("refresh", "1");
      const data = await getJSON(`/api/calendar?${qs.toString()}`);
      lastCalendarData = data;
      renderPlan(data);
      renderBias(data);
      renderCalendar(data);
      renderNewsBanner(data);
      lastCalendarLoad = Date.now();
    } catch (e) {
      const sub = document.getElementById("calendar-sub");
      if (sub) sub.textContent = `calendar error: ${e.message}`;
    }
  }

  function initCalendarControls() {
    const seg = document.querySelectorAll(".seg-btn[data-impact]");
    if (!seg.length) return;
    seg.forEach((btn) => {
      btn.addEventListener("click", () => {
        const imp = btn.dataset.impact || "High,Medium";
        if (imp === calendarImpact) return;
        calendarImpact = imp;
        seg.forEach((b) => b.classList.toggle("is-active", b === btn));
        refreshCalendar(false);
      });
    });
  }

  // ----------------------------------------------------------------- timeframe picker
  function _reloadChartForTimeframe() {
    if (chart && candleSeries) {
      chart.removeSeries(candleSeries);
      candleSeries = chart.addCandlestickSeries({
        upColor: theme.win, downColor: theme.loss,
        borderUpColor: theme.win, borderDownColor: theme.loss,
        wickUpColor: theme.win, wickDownColor: theme.loss,
      });
    }
    lastCandleTime = 0;
    loadCandlesFull();
  }

  async function initTimeframePicker() {
    const buttons = Array.from(document.querySelectorAll(".seg-btn[data-tf]"));
    if (!buttons.length) return;

    // First: figure out the default TF from /api/status so the highlight
    // matches the bot's actual configured timeframe on first paint. The
    // user's localStorage choice always wins over the default.
    let defaultTf = "M5";
    try {
      const s = await getJSON(withSymbol("/api/status"));
      if (s && s.timeframe) defaultTf = s.timeframe;
    } catch (_) { /* fall back to M5 */ }
    if (!currentTimeframe) currentTimeframe = defaultTf;

    const setActive = (tf) => {
      buttons.forEach((b) => b.classList.toggle("is-active", b.dataset.tf === tf));
    };
    setActive(currentTimeframe);

    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const tf = btn.dataset.tf;
        if (!tf || tf === currentTimeframe) return;
        currentTimeframe = tf;
        try { localStorage.setItem("rt:tf", currentTimeframe); } catch (_) {}
        setActive(tf);
        _reloadChartForTimeframe();
      });
    });
  }

  // ----------------------------------------------------------------- symbol picker
  /** Read ``?symbols=A,B,C`` (or hash equivalent) from the page URL. Lets you
   * pin pairs into the picker that haven't traded yet (useful while waiting for
   * the first signal on a freshly-started bot). */
  function readSeedSymbols() {
    try {
      const qs = new URLSearchParams(window.location.search);
      const fromQuery = qs.get("symbols");
      if (fromQuery) return fromQuery;
      // Also support #symbols=... in case the user prefers it
      const hash = (window.location.hash || "").replace(/^#/, "");
      const hashParams = new URLSearchParams(hash);
      return hashParams.get("symbols") || "";
    } catch (_) {
      return "";
    }
  }

  async function initSymbolPicker() {
    const sel = document.getElementById("symbol-select");
    if (!sel) return;
    const seed = readSeedSymbols();
    const url = seed ? `/api/symbols?seed=${encodeURIComponent(seed)}` : "/api/symbols";
    try {
      const r = await getJSON(url);
      knownSymbols = r.symbols || [];
      // If localStorage had a stale symbol the bot no longer reports, fall back
      // to the dashboard's configured default.
      if (!currentSymbol || !knownSymbols.includes(currentSymbol)) {
        currentSymbol = r.default || (knownSymbols[0] || "");
      }
      sel.innerHTML = "";
      knownSymbols.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        if (s === currentSymbol) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener("change", () => {
        currentSymbol = sel.value;
        try { localStorage.setItem("rt:symbol", currentSymbol); } catch (_) {}
        // Force a fresh chart on symbol switch: dispose the old series so
        // setData below replays the candle layout cleanly.
        if (chart && candleSeries) {
          chart.removeSeries(candleSeries);
          candleSeries = chart.addCandlestickSeries({
            upColor: theme.win, downColor: theme.loss,
            borderUpColor: theme.win, borderDownColor: theme.loss,
            wickUpColor: theme.win, wickDownColor: theme.loss,
          });
        }
        lastCandleTime = 0;
        loadCandlesFull();
        refreshCalendar(false);
        tick();
      });
      // Re-poll the symbol list periodically so newly-active pairs appear
      // without a page reload. Cheap query against the journal.
      setInterval(async () => {
        try {
          const rr = await getJSON(url);
          const fresh = rr.symbols || [];
          if (fresh.length === knownSymbols.length &&
              fresh.every((v, i) => v === knownSymbols[i])) return;
          knownSymbols = fresh;
          // Repopulate the dropdown without dropping the current selection
          const cur = currentSymbol;
          sel.innerHTML = "";
          knownSymbols.forEach((s) => {
            const opt = document.createElement("option");
            opt.value = s;
            opt.textContent = s;
            if (s === cur) opt.selected = true;
            sel.appendChild(opt);
          });
        } catch (_) { /* transient */ }
      }, 30_000);
    } catch (e) {
      console.warn("symbol picker init failed:", e);
    }
  }

  // ----------------------------------------------------------------- main loop
  async function tick() {
    await Promise.all([
      refreshStatus(),
      refreshStats(),
      refreshPositions(),
      refreshSignals(),
      refreshTrades(),
      refreshLastCandle(),
      refreshEquityCurve(),
    ]);
    if (Date.now() - lastChartLoad > CHART_REFRESH_MS) {
      // Reload candles + markers periodically (cheap)
      loadCandlesFull();
    }
    if (Date.now() - lastNewsLoad > NEWS_REFRESH_MS) {
      // News are server-cached for 5 min, but we still pull more often so
      // the relative-age labels stay fresh and the user sees new items soon
      // after the cache expires.
      refreshNews(false);
    }
    if (Date.now() - lastCalendarLoad > CALENDAR_REFRESH_MS) {
      refreshCalendar(false);
    }
    setText("last-update", new Date().toISOString().replace("T", " ").slice(11, 19) + "Z");
  }

  document.addEventListener("DOMContentLoaded", async () => {
    const secs = String(REFRESH_MS / 1000);
    setText("refresh-secs", secs);
    setText("refresh-secs-foot", secs);
    initNewsControls();
    initCalendarControls();
    await initSymbolPicker();
    await initTimeframePicker();
    loadCandlesFull();
    refreshNews(false);
    refreshCalendar(false);
    tick();
    setInterval(tick, REFRESH_MS);
    setInterval(tickCountdowns, COUNTDOWN_MS);
  });
})();
