/* Robot Trading dashboard - vanilla JS */
(function () {
  "use strict";

  const REFRESH_MS = 5000;
  const CHART_REFRESH_MS = 15000;

  let chart = null;
  let candleSeries = null;
  let lastCandleTime = 0;
  let lastChartLoad = 0;

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

  // ----------------------------------------------------------------- chart
  function ensureChart() {
    if (chart) return;
    const el = document.getElementById("chart");
    chart = LightweightCharts.createChart(el, {
      layout: { background: { color: "#161b22" }, textColor: "#e6edf3" },
      grid: { vertLines: { color: "#2a313c" }, horzLines: { color: "#2a313c" } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: "#2a313c" },
      rightPriceScale: { borderColor: "#2a313c" },
      crosshair: { mode: 1 },
      autoSize: true,
    });
    candleSeries = chart.addCandlestickSeries({
      upColor: "#26a69a", downColor: "#ef5350",
      borderUpColor: "#26a69a", borderDownColor: "#ef5350",
      wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    });
    window.addEventListener("resize", () => chart && chart.applyOptions({}));
  }

  async function loadCandlesFull() {
    ensureChart();
    try {
      const data = await getJSON("/api/candles?bars=500");
      if (data.error) {
        setText("chart-sub", `chart unavailable: ${data.error}`);
        return;
      }
      candleSeries.setData(data.candles);
      lastCandleTime = data.candles.length ? data.candles[data.candles.length - 1].time : 0;
      setText("chart-title", `${data.symbol} (${data.timeframe})`);
      setText("chart-sub", `${data.candles.length} bars`);
      // Markers
      const m = await getJSON("/api/chart_markers?limit=200");
      if (m.markers) candleSeries.setMarkers(m.markers);
      lastChartLoad = Date.now();
    } catch (e) {
      setText("chart-sub", `chart error: ${e.message}`);
    }
  }

  async function refreshLastCandle() {
    if (!candleSeries) return;
    try {
      const data = await getJSON("/api/candles?bars=2");
      if (!data.candles || !data.candles.length) return;
      data.candles.forEach((c) => {
        if (c.time >= lastCandleTime) {
          candleSeries.update(c);
          lastCandleTime = c.time;
        }
      });
    } catch (_) { /* ignore transient errors */ }
  }

  // ----------------------------------------------------------------- top bar / kpis
  async function refreshStatus() {
    try {
      const s = await getJSON("/api/status");
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
      const s = await getJSON("/api/stats");
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
      const r = await getJSON("/api/positions");
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
      const r = await getJSON("/api/signals?limit=20");
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
      const r = await getJSON("/api/trades?limit=30");
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

  // ----------------------------------------------------------------- main loop
  async function tick() {
    await Promise.all([
      refreshStatus(),
      refreshStats(),
      refreshPositions(),
      refreshSignals(),
      refreshTrades(),
      refreshLastCandle(),
    ]);
    if (Date.now() - lastChartLoad > CHART_REFRESH_MS) {
      // Reload candles + markers periodically (cheap)
      loadCandlesFull();
    }
    setText("last-update", new Date().toISOString().replace("T", " ").slice(11, 19) + "Z");
  }

  document.addEventListener("DOMContentLoaded", () => {
    setText("refresh-secs", String(REFRESH_MS / 1000));
    loadCandlesFull();
    tick();
    setInterval(tick, REFRESH_MS);
  });
})();
