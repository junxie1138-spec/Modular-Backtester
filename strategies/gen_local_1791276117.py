from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueAnalogParams:
    lookback: int = 126
    trail_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[QueueAnalogParams]):
    strategy_id = "gen_local_1791276117"

    @classmethod
    def params_type(cls) -> type[QueueAnalogParams]:
        return QueueAnalogParams

    def warmup_bars(self, params: QueueAnalogParams) -> int:
        return int(params.lookback) + 12

    def indicators(self, data: pd.DataFrame, params: QueueAnalogParams) -> pd.DataFrame:
        if params.lookback < 14 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("lookback must be >= 14 and trail_atr must be positive")
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame({"return": returns}, index=data.index)
        # Reflect cumulative selling pressure at zero: gains drain the loss queue.
        path = np.column_stack([returns.shift(lag).to_numpy() for lag in range(4, -1, -1)])
        cumulative = np.cumsum(-path, axis=1)
        floor = np.minimum(0.0, np.minimum.accumulate(cumulative, axis=1))
        scale = np.abs(path).sum(axis=1)
        scale[scale == 0] = np.nan
        shape = (cumulative - floor) / scale[:, None]
        columns = [f"queue_{j}" for j in range(5)]
        out[columns] = shape
        previous = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        n = len(data)
        if n == 0:
            out["forecast"] = pd.Series(index=data.index, dtype=float)
            out["agreement"] = pd.Series(index=data.index, dtype=float)
            return out
        distances = np.full((n, params.lookback), np.inf)
        outcomes = np.zeros((n, params.lookback))
        for slot, lag in enumerate(range(7, params.lookback + 7)):
            historical = out[columns].shift(lag).to_numpy()
            distance = np.square(shape - historical).sum(axis=1)
            outcome = (close.shift(lag - 2) / close.shift(lag) - 1).to_numpy()
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[valid, slot] = distance[valid]
            outcomes[valid, slot] = outcome[valid]
        nearest = np.argpartition(distances, 6, axis=1)[:, :7]
        selected = np.take_along_axis(outcomes, nearest, axis=1)
        enough = np.isfinite(distances).sum(axis=1) >= 7
        out["forecast"] = np.where(enough, selected.mean(axis=1), 0.0)
        out["agreement"] = np.where(enough, (selected > 0).mean(axis=1), 0.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: QueueAnalogParams
    ) -> SignalFrame:
        eligible = (
            (indicators["forecast"] > 0.0014)
            & (indicators["agreement"] >= 5 / 7)
            & (indicators["queue_4"] > 0)
            & (indicators["queue_4"] < indicators["queue_3"])
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy()
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry = stop = peak = 0.0
        armed = False
        age = 0
        for i, (opening, high, low, closing) in enumerate(prices):
            valid = np.isfinite([opening, high, low, closing]).all() and min(opening, low, closing) > 0
            if pending:
                pending = False
                if not valid or i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    active = False
                    continue
                # The preceding close requested entry; this bar's open is the fill proxy.
                entry = peak = opening
                stop = entry - 1.5 * atr[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                if not valid or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                hit = opening <= stop or low <= stop
                peak = max(peak, high)
                armed = armed or peak >= entry * 1.004
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # Prior stop applies to the low; the newly raised stop applies to the close.
                # Every exit is a close decision executed at the following open.
                if hit or closing <= stop or age >= 2:
                    active = False
                else:
                    signal[i] = 1
            elif valid and eligible[i]:
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = QueueAnalogParams(lookback=63)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100, 100, 101, 102, 101, 100],
        "high": [100, 100.2, 104, 103, 102, 101],
        "low": [99, 99, 100.5, 101, 100, 99],
        "close": [100, 100.1, 103, 102, 101, 100],
        "volume": 1000
    }, index=index)
    ind = pd.DataFrame({
        "forecast": [0.01, 0, 0, 0, 0, 0], "agreement": 1.0,
        "queue_3": 0.5, "queue_4": 0.25, "atr": 2.0
    }, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    stopped = data.copy()
    stopped.loc[index[1], "high"] = 101
    stopped.loc[index[1], "close"] = 99.9
    assert strategy.generate_signals(stopped, ind, None, params).data["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
    history = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000
    }, index=pd.date_range("2021-01-01", periods=100))
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:90], params)
    pd.testing.assert_frame_equal(full.iloc[:90], prefix)


if __name__ == "__main__":
    _self_check()
