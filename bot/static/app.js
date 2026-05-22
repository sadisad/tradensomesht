/* Robot Trading dashboard - vanilla JS */
(function () {
  "use strict";

  const REFRESH_MS = 5000;
  const CHART_REFRESH_MS = 15000;

  let chart = null;
  let candleSeries = null;
  let equityChart = null;
  let equitySeries = null;
  let lastCandleTime = 0;
  let lastChartLoad = 0;

  // Currently-selected symbol. Initialised from localStorage on boot, then
  // overwritten once /api/symbols replies with the canonical list.
  let currentSymbol = (typeof localStorage !== "undefined"
    ? localStorage.getItem("rt:symbol")
    : null) || "";
  let knownSymbols = [];

  // ---- Theme: pull live values from CSS so charts match the rest of the UI.
  // Reading them once at boot is fine because the theme is static per page load.
  const css = getComputedStyle(document.documentElement);
  const theme = {
    bg:      (css.getPropertyValue("--bg-chart")    || "#FFF6E5").trim(),
    text:    (css.getPropertyValue("--text")        || "#1a2332").trim(),
    grid:    (css.getPropertyValue("--line-soft")   || "rgba(140,192,235,0.28)").trim(),
    border:  (css.getPropertyValue("--line")        || "#BFDDF0").trim(),
    win:     (css.getPropertyValue("--win")         || "#0d8a6e").trim(),
    loss:    (css.getPropertyValue("--loss")        || "#b8423a").trim(),
    accent:  (css.getPropertyValue("--accent-strong") || "#5FA3D6").trim(),
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
      const data = await getJSON(withSymbol("/api/candles?bars=500"));
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
      const data = await getJSON(withSymbol("/api/candles?bars=2"));
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
      topColor: "rgba(95, 163, 214, 0.30)",
      bottomColor: "rgba(95, 163, 214, 0.02)",
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

  // ----------------------------------------------------------------- symbol picker
  async function initSymbolPicker() {
    const sel = document.getElementById("symbol-select");
    if (!sel) return;
    try {
      const r = await getJSON("/api/symbols");
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
        tick();
      });
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
    setText("last-update", new Date().toISOString().replace("T", " ").slice(11, 19) + "Z");
  }

  document.addEventListener("DOMContentLoaded", async () => {
    setText("refresh-secs", String(REFRESH_MS / 1000));
    await initSymbolPicker();
    loadCandlesFull();
    tick();
    setInterval(tick, REFRESH_MS);
  });
})();
