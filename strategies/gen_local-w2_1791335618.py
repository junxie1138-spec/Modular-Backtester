from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_window: int = 32
    fd_order: float = 0.45
    percentile_window: int = 63
    entry_percentile: float = 0.75
    atr_window: int = 14
    stop_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791335618"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fd_window + params.percentile_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.fd_window >= 2 and params.percentile_window >= 2
                and params.atr_window >= 1 and 0.0 < params.fd_order < 1.0
                and 0.5 < params.entry_percentile < 1.0
                and params.stop_atr > 0.0 and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["log_close"] = np.log(close.where(close > 0.0))
        weights = np.ones(params.fd_window, dtype=float)
        for lag in range(1, params.fd_window):
            weights[lag] = weights[lag - 1] * (lag - 1 - params.fd_order) / lag
        # Fixed-width fractional difference: every weight uses a past close.
        fd = np.full(len(data), np.nan, dtype=float)
        if len(data):
            fd = np.convolve(out["log_close"].to_numpy(), weights, mode="full")[:len(data)]
            fd[:params.fd_window - 1] = np.nan
        out["fd_close"] = fd
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        history = out["fd_close"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out["fd_upper"] = history.quantile(params.entry_percentile)
        price_history = close.shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out["price_median"] = price_history.quantile(0.5)
        out["price_upper"] = price_history.quantile(params.entry_percentile)
        # Memory leads: fractional close escapes its upper percentile while
        # the actual close is still inside the upper half of its price range.
        out["entry"] = (
            (out["fd_close"] > out["fd_upper"])
            & (out["fd_close"].shift(1) <= out["fd_upper"].shift(1))
            & (close > out["price_median"])
            & (close < out["price_upper"])
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        peak = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0.0
            valid = valid and np.isfinite(atrs[i]) and atrs[i] > 0.0
            if active:
                age += 1
                if valid:
                    # The first held close is on the next bar after entry.
                    peak = max(peak, closes[i])
                    stop = max(stop, peak - params.stop_atr * atrs[i])
                if not valid or closes[i] <= stop or age >= params.hold_bars:
                    active = False
            elif valid and entries[i]:
                active = True
                age = 0
                peak = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    closes = np.array([100., 101., 105., 104., 103., 102., 101., 100.])
    data = pd.DataFrame({"open": closes, "high": closes + 1.0,
                         "low": closes - 1.0, "close": closes,
                         "volume": 1000.0}, index=index)
    fixture = pd.DataFrame({"atr": [1., 1., 1., 4., 4., 4., 4., 4.],
                            "entry": [True] + [False] * 7}, index=index)
    # ATR expansion must not loosen the stop established at the 105 close.
    result = strategy.generate_signals(data, fixture, None, Params(hold_bars=5)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    rising = data.copy()
    rising["close"] = np.arange(100., 108.)
    result = strategy.generate_signals(rising, fixture, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0.0).all()
    empty = data.iloc[:0]
    assert strategy.indicators(empty, Params()).empty
