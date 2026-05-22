# Robot Trading

Automated XAUUSD / forex trading bot for MetaTrader 5. Pulls OHLCV in real time,
generates EMA + RSI + ATR signals, filters them with a self-improving ML model,
sizes positions from ATR risk, and journals every trade so the model gets
better the more it trades.

> ## Reality check, read this first
>
> No bot makes money out of the box. This one is a sound *framework* with
> conservative defaults, not a printing press. Expect to spend real time
> backtesting and tuning on demo before risking a cent. Live with size only
> after the strategy shows positive expectancy on out-of-sample data.

## Features

- MetaTrader 5 integration (data + orders, magic-number-tagged)
- Strategy: EMA(20/50/200) trend + RSI(14) momentum + ATR-based stops
- Risk: % of equity per trade, ATR-based SL/TP, dynamic risk scaling from rolling win-rate
- ML filter: HistGradientBoostingClassifier predicts win probability; signals below threshold are skipped
- Self-improvement: every closed trade is journaled, model retrains every N trades on the live history
- Safety guards: trading-hours window, post-loss cooldown, daily loss kill-switch, max open positions, paper mode
- Vectorised backtester for offline strategy validation
- SQLite trade journal + CSV export
- Live web dashboard (FastAPI + lightweight-charts) to watch the bot in real time

## Project layout

```
config.yaml          # all knobs live here
requirements.txt
bot/
  __init__.py
  config.py          # YAML loader
  logging_setup.py   # rotating file + stream logger
  broker_mt5.py      # MT5 facade
  indicators.py      # EMA / RSI / ATR + ML feature builder
  strategy.py        # EMA+RSI signal rules
  risk.py            # sizing, SL/TP, gates
  journal.py         # SQLite trade journal
  ml_filter.py       # train / predict / retrain
  backtest.py        # vectorised backtester
  live.py            # the live loop (entrypoint)
  run_backtest.py    # backtest CLI
  tools.py           # retrain / export / stats CLI
  dashboard.py       # FastAPI dashboard server
  static/            # HTML / CSS / JS for the dashboard
data/                # journal db, logs, csv exports (gitignored)
models/              # trained ML model (gitignored)
```

## Setup (Windows, MT5)

1. Install the [MetaTrader 5 terminal](https://www.metatrader5.com/) and create
   a **demo account**. Log in inside the terminal at least once.
2. Install Python 3.10+ (3.11 recommended).
3. From the project folder:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. Edit `config.yaml`:
   - `broker.login` / `password` / `server` (your demo creds)
   - `broker.magic` (any unique int; tags this bot's orders)
   - `trading.symbol` (e.g. `XAUUSD`, may differ by broker e.g. `XAUUSD.s`)
   - Leave `trading.mode: paper` for the first run

> Never commit real credentials. `.gitignore` already excludes `config.local.yaml`
> and `.env` if you prefer to keep secrets there.

## Run

### Paper mode (no real orders, synthetic fills against live bars)

```powershell
python -m bot.live
```

This connects to MT5 (read-only), watches new bars, logs signals, and simulates
opens/closes against live OHLCV. Every trade goes into `data/journal.db`. Once
you have ~`ml.min_train_samples` closed paper trades, the ML filter trains and
starts gating signals.

### Demo mode (places real orders on your demo account)

Set in `config.yaml`:

```yaml
trading:
  mode: demo
```

Then `python -m bot.live` again. Orders are sent with the configured magic
number; the bot only sees / closes its own positions.

### Backtest

```powershell
# Pull recent history from the connected MT5 terminal
python -m bot.run_backtest --from-mt5 --bars 30000 --out data/bt_trades.csv

# Or from a CSV (columns: time,open,high,low,close,volume; time is UTC)
python -m bot.run_backtest --csv data/xauusd_m5.csv
```

### Tools

```powershell
python -m bot.tools stats               # journal summary
python -m bot.tools retrain             # force ML retrain from journal
python -m bot.tools export --out data/trades.csv
```

### Live dashboard

Open a second terminal (the bot keeps running in the first one):

```powershell
python -m bot.dashboard
```

Then open http://127.0.0.1:8765 in a browser. You'll see:

- Connection status, account equity / balance, current bid/ask
- Live candlestick chart of the configured symbol/timeframe
- Trade entry/exit markers overlaid on the chart (green = buy / win, red = sell / loss)
- Open positions table with live PnL (demo mode pulls from MT5; paper mode reads the journal)
- Recent signals feed (acted vs skipped, with skip reason and ML probability)
- Recent closed trades with PnL and W/L
- 24h activity stats and overall win-rate

The dashboard polls every 5 seconds, refreshes the full chart every 15 seconds,
and runs read-only against MT5 + the journal. It's safe to leave open while the
bot trades. Bind to a non-loopback host carefully:

```powershell
python -m bot.dashboard --host 0.0.0.0 --port 8765   # exposes to your LAN
```

## How the self-improvement loop works

1. Strategy fires a signal on the last closed bar.
2. Features at that bar (returns over multiple horizons, EMA slopes, RSI,
   ATR%, time-of-day, etc.) are snapshotted.
3. `MLFilter.predict_proba_win(features)` returns the model's confidence.
   If below `ml.min_proba_to_trade` the signal is recorded as `acted=0` with
   `skip_reason=ml_proba_below_threshold` and ignored.
4. If accepted, the risk manager scales risk by rolling win-rate of the same
   side (`risk.risk_scale_from_history`), builds SL/TP/volume, and the order
   goes out (or a paper trade is opened).
5. When the position closes, `Journal.record_close` writes the outcome.
6. Every `ml.retrain_every_n_trades` closed trades, the model retrains on the
   full journal. Time-series CV metrics (AUC / accuracy / log-loss) are logged.

The result is a bot that progressively skips setups that historically lost and
sizes up setups that historically won, without changing the underlying rules
out from under itself mid-trade.

## Things this bot deliberately does not do

- No martingale / grid / averaging-down. These blow up.
- No news scraping. Easy to add later via a feature function in `indicators.py`.
- No multi-symbol portfolio logic. One symbol per process keeps state simple.
- No leverage tweaking. Use your broker's account-level leverage; risk is
  controlled by lot size off the SL distance.

## Tuning checklist before going live

- Backtest on >1 year of `XAUUSD` M5 data, ideally across regimes (2022 trend,
  2023 chop, 2024 etc.).
- Confirm `avg_R > 0` and `max_consec_losses` is survivable at your risk %.
- Run paper mode for at least 2 weeks. Inspect `bot.tools stats`.
- Move to demo mode for another 2-4 weeks.
- Only then consider a live account, with `risk.risk_per_trade_pct` halved.

## Troubleshooting

- "MetaTrader5 package is not installed": `pip install MetaTrader5` (Windows only).
- "mt5.initialize() failed": the terminal isn't running, or `broker.path` is
  wrong. Open the MT5 terminal manually first.
- "Symbol not visible": your broker may use a suffix (e.g. `XAUUSD.s`,
  `XAUUSD-pro`). Update `trading.symbol`.
- `cv_auc` printed as `nan`: not enough samples or one class missing. Trade
  longer in paper mode before forcing a retrain.
