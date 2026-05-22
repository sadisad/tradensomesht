"""MetaTrader 5 broker client.

Thin wrapper around the official ``MetaTrader5`` Python package. Responsibilities:

* connect / disconnect to the terminal
* fetch OHLCV history as a pandas DataFrame
* fetch latest tick / current price
* place market orders with SL/TP
* close an existing position
* list this bot's open positions (filtered by magic number)

We intentionally avoid leaking ``MetaTrader5`` types outside this module: the
strategy / risk / journal layers should stay testable without the real terminal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .logging_setup import get_logger

log = get_logger(__name__)

try:  # pragma: no cover - import guarded so non-Windows dev machines can lint
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # noqa: BLE001
    mt5 = None  # filled in at runtime; we error loudly if the user tries to connect


_TF_MAP_NAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


@dataclass
class Position:
    ticket: int
    symbol: str
    side: str            # "buy" or "sell"
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    time_open: datetime
    magic: int
    comment: str


@dataclass
class OrderResult:
    ok: bool
    ticket: Optional[int]
    price: Optional[float]
    retcode: Optional[int]
    comment: str


class MT5Client:
    """Lightweight MT5 facade. One instance per process."""

    def __init__(self, broker_cfg: Dict[str, Any]):
        self.cfg = broker_cfg
        self.magic: int = int(broker_cfg.get("magic", 0))
        self._connected = False

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 package is not installed. Run `pip install MetaTrader5` "
                "on a Windows machine with the MT5 terminal installed."
            )
        path = self.cfg.get("path") or None
        init_kwargs: Dict[str, Any] = {}
        if path:
            init_kwargs["path"] = path

        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            raise RuntimeError(f"mt5.initialize() failed: {err}")

        login = int(self.cfg.get("login") or 0)
        password = self.cfg.get("password") or ""
        server = self.cfg.get("server") or ""
        if login and password and server:
            ok = mt5.login(login=login, password=password, server=server)
            if not ok:
                err = mt5.last_error()
                mt5.shutdown()
                raise RuntimeError(f"mt5.login() failed: {err}")
        else:
            log.warning(
                "MT5 login/password/server not set in config. Using whatever account "
                "the terminal is currently logged into."
            )

        info = mt5.account_info()
        if info is None:
            raise RuntimeError("mt5.account_info() returned None after login")
        log.info(
            "MT5 connected: login=%s server=%s balance=%.2f equity=%.2f currency=%s",
            info.login, info.server, info.balance, info.equity, info.currency,
        )
        self._connected = True

    def disconnect(self) -> None:
        if mt5 is not None and self._connected:
            mt5.shutdown()
        self._connected = False

    # ------------------------------------------------------------------ account
    def account_equity(self) -> float:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError("account_info() returned None")
        return float(info.equity)

    def account_balance(self) -> float:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError("account_info() returned None")
        return float(info.balance)

    def symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None or not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Failed to select symbol {symbol}")
            info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) returned None")
        return info

    # ------------------------------------------------------------------ market data
    def get_rates(self, symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
        """Return last ``n_bars`` OHLCV rows, oldest first, indexed by UTC time."""
        tf_name = _TF_MAP_NAMES.get(timeframe.upper())
        if tf_name is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        tf = getattr(mt5, tf_name)
        self.symbol_info(symbol)  # ensure visible
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(n_bars))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos returned no data for {symbol} {timeframe}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]]

    def current_price(self, symbol: str, side: str) -> float:
        """Best price to fill ``side`` (buy => ask, sell => bid)."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) returned None")
        return float(tick.ask if side == "buy" else tick.bid)

    def current_spread_points(self, symbol: str) -> float:
        """Current bid-ask spread expressed in *points* (1/point per symbol_info)."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) returned None")
        info = self.symbol_info(symbol)
        point = float(getattr(info, "point", 0.0)) or 0.00001
        return float(tick.ask - tick.bid) / point

    def current_spread_price(self, symbol: str) -> float:
        """Current bid-ask spread in raw price units."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) returned None")
        return float(tick.ask - tick.bid)

    # ------------------------------------------------------------------ orders
    def place_market_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "robot",
        deviation: int = 20,
    ) -> OrderResult:
        info = self.symbol_info(symbol)
        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        price = self.current_price(symbol, side)

        # Pick a filling mode the broker supports. IOC is safest for market orders.
        filling = mt5.ORDER_FILLING_IOC
        try:
            allowed = int(getattr(info, "filling_mode", 0))
            if allowed and not (allowed & 1):  # 1 == FOK, 2 == IOC, 4 == RETURN; bit-mask varies by build
                filling = mt5.ORDER_FILLING_FOK
        except Exception:  # noqa: BLE001
            pass

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(deviation),
            "magic": int(self.magic),
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            log.error("order_send returned None: %s", err)
            return OrderResult(False, None, None, None, f"order_send None: {err}")

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            log.error(
                "order_send failed: retcode=%s comment=%s request=%s",
                result.retcode, result.comment, request,
            )
        else:
            log.info(
                "order placed: %s %s vol=%.2f price=%.5f sl=%.5f tp=%.5f ticket=%s",
                side, symbol, volume, result.price, sl, tp, result.order,
            )
        return OrderResult(
            ok=ok,
            ticket=int(result.order) if ok else None,
            price=float(result.price) if ok else None,
            retcode=int(result.retcode),
            comment=str(result.comment),
        )

    def close_position(self, position: Position, deviation: int = 20) -> OrderResult:
        opp_side = "sell" if position.side == "buy" else "buy"
        order_type = mt5.ORDER_TYPE_SELL if position.side == "buy" else mt5.ORDER_TYPE_BUY
        price = self.current_price(position.symbol, opp_side)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": order_type,
            "position": int(position.ticket),
            "price": price,
            "deviation": int(deviation),
            "magic": int(self.magic),
            "comment": "robot-close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            return OrderResult(False, None, None, None, f"order_send None: {err}")
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            ok=ok,
            ticket=int(position.ticket),
            price=float(result.price) if ok else None,
            retcode=int(result.retcode),
            comment=str(result.comment),
        )

    def modify_position_sltp(
        self,
        position: Position,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        """Modify SL/TP for an existing position. Pass ``None`` to leave a value unchanged.

        Returns OrderResult with ok=True on success. The broker rejects changes that
        violate stops_level (e.g. SL too close to market); callers should check ``ok``.
        """
        new_sl = float(sl) if sl is not None else float(position.sl)
        new_tp = float(tp) if tp is not None else float(position.tp)
        if new_sl == position.sl and new_tp == position.tp:
            # No-op; report as success without round-tripping the broker
            return OrderResult(True, int(position.ticket), None, 0, "noop")
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": int(position.ticket),
            "sl": new_sl,
            "tp": new_tp,
            "magic": int(self.magic),
        }
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            log.error("modify_position_sltp returned None: %s", err)
            return OrderResult(False, None, None, None, f"order_send None: {err}")
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            log.warning(
                "modify_position_sltp failed: ticket=%s retcode=%s comment=%s sl=%.5f tp=%.5f",
                position.ticket, result.retcode, result.comment, new_sl, new_tp,
            )
        return OrderResult(
            ok=ok,
            ticket=int(position.ticket),
            price=None,
            retcode=int(result.retcode),
            comment=str(result.comment),
        )

    # ------------------------------------------------------------------ positions
    def open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        out: List[Position] = []
        for p in positions:
            if int(p.magic) != int(self.magic):
                continue  # ignore positions opened by humans / other EAs
            out.append(
                Position(
                    ticket=int(p.ticket),
                    symbol=str(p.symbol),
                    side="buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                    volume=float(p.volume),
                    price_open=float(p.price_open),
                    sl=float(p.sl),
                    tp=float(p.tp),
                    profit=float(p.profit),
                    time_open=datetime.fromtimestamp(p.time, tz=timezone.utc),
                    magic=int(p.magic),
                    comment=str(p.comment),
                )
            )
        return out

    # ------------------------------------------------------------------ history (closed deals)
    def deals_since(self, since: datetime) -> List[Dict[str, Any]]:
        """Return raw closed-deal dicts since ``since`` (UTC). Used to detect
        positions closed by SL/TP between loop iterations."""
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        to = datetime.now(tz=timezone.utc)
        deals = mt5.history_deals_get(since, to)
        if deals is None:
            return []
        out = []
        for d in deals:
            if int(d.magic) != int(self.magic):
                continue
            out.append({
                "ticket": int(d.ticket),
                "order": int(d.order),
                "position_id": int(d.position_id),
                "symbol": str(d.symbol),
                "type": int(d.type),  # 0 buy, 1 sell
                "entry": int(d.entry),  # 0 in, 1 out, 2 inout
                "volume": float(d.volume),
                "price": float(d.price),
                "profit": float(d.profit),
                "time": datetime.fromtimestamp(d.time, tz=timezone.utc),
                "comment": str(d.comment),
            })
        return out
