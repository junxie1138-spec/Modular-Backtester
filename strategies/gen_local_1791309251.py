from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    return_window: int = 20
    price_window: int = 20
    atr_window: int = 14
    return_tail: float = 0.15
    price_floor: float = 0.65
    stop_atr: float = 1.0
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791309251"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.return_window + 2, params.price_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.return_window, params.price_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 <= params.return_tail < 1 and 0 < params.price_floor <= 1):
            raise ValueError("Relative-position thresholds must be in range.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0 or params.hold_days not in (1, 2):
            raise ValueError("Use a positive stop and a one- or two-day holding period.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["return"] = close.pct_change(fill_method=None)
        history = out["return"].shift(1).rolling(params.return_window)
        out["return_low"] = history.min()
        out["return_high"] = history.max()
        return_width = (out["return_high"] - out["return_low"]).replace(0, np.nan)
        out["return_position"] = ((out["return"] - out["return_low"]) / return_width).clip(0, 1)
        prices = previous.rolling(params.price_window)
        out["price_low"] = prices.min()
        out["price_high"] = prices.max()
        price_width = (out["price_high"] - out["price_low"]).replace(0, np.nan)
        out["price_position"] = ((close - out["price_low"]) / price_width).clip(0, 1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        # Primitive 1: a negative return near the bottom of its historical range.
        return_vote = (indicators["return"] < 0) & (indicators["return_position"] <= params.return_tail)
        # Primitive 2: the close still occupies the upper part of its price range.
        price_vote = indicators["price_position"] >= params.price_floor
        entry = (return_vote & price_vote & (indicators["atr"] > 0)).fillna(False).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The previous close scheduled this entry; this open is now known.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                if (not np.isfinite(stop) or not np.isfinite(closes[i])
                        or closes[i] < stop or i - entry_bar + 1 >= params.hold_days):
                    active = False
                else:
                    raw[i] = 1
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]):
                active = True
                entry_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100., 110., 100., 100., 100., 100., 100.],
        "high": [101., 111., 101., 101., 101., 101., 101.],
        "low": [99., 104., 99., 99., 99., 99., 99.],
        "close": [100., 105., 100., 100., 100., 100., 100.],
        "volume": 1000.,
    }, index=index)
    ind = pd.DataFrame({
        "return": [-0.01, 0., -0.01, -0.01, 0., 0., 0.],
        "return_position": [0., 1., 0., 0., 1., 1., 1.],
        "price_position": [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9],
        "atr": [2., 100., 2., 2., 2., 2., 2.],
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, Params()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    one_day = strategy.generate_signals(data, ind, ctx, Params(hold_days=1)).data
    assert one_day["signal"].tolist() == [0, 1, 0, 0, 1, 0, 0]
    flat = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                         "close": 100., "volume": 1000.},
                        index=pd.date_range("2020-01-01", periods=60))
    flat_signals = strategy.generate_signals(flat, strategy.indicators(flat, Params()), ctx, Params()).data
    assert flat_signals["signal"].eq(0).all()
    assert result.index.equals(data.index) and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
