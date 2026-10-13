from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 12
    stop_atr: float = 1.8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791315725"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.graph_window, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 4 <= params.graph_window <= 64:
            raise ValueError("graph_window must be between 4 and 64")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["atr"] = tr.rolling(14, min_periods=14).mean()
        out["topology"] = 0.0
        w = params.graph_window
        if len(data) < w:
            return out
        values = np.log(close.where(close > 0)).to_numpy()
        windows = np.lib.stride_tricks.sliding_window_view(values, w)
        valid = np.isfinite(windows).all(axis=1)
        count = np.zeros(windows.shape)
        signed = np.zeros(windows.shape)
        # ponytail: O(bars * window squared); cap at 64, chunk rows for large histories.
        for a in range(w - 2):
            slopes = (windows[:, a + 1:] - windows[:, a, None]) / np.arange(1, w - a)
            upper = np.maximum.accumulate(slopes, axis=1)
            lower = np.minimum.accumulate(slopes, axis=1)
            visible = (
                (slopes[:, 1:] > upper[:, :-1] + 1e-12)
                | (slopes[:, 1:] < lower[:, :-1] - 1e-12)
            )
            for offset in range(visible.shape[1]):
                b = a + offset + 2
                edge = visible[:, offset] & valid
                direction = np.sign(windows[:, b] - windows[:, a])
                flow = np.where(edge, direction, 0.0)
                count[:, a] += edge
                count[:, b] -= edge
                signed[:, a] += flow
                signed[:, b] -= flow
        capacity = np.cumsum(count, axis=1)[:, :-1]
        net = np.cumsum(signed, axis=1)[:, :-1]
        balance = np.divide(net, capacity, out=np.zeros_like(net), where=capacity > 0)
        weakest_up = balance.min(axis=1)
        weakest_down = balance.max(axis=1)
        score = np.where(weakest_up > 0, weakest_up,
                         np.where(weakest_down < 0, weakest_down, 0.0))
        score[~valid] = 0.0
        out.iloc[w - 1:, out.columns.get_loc("topology")] = score
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        score = indicators["topology"].to_numpy()
        atr = indicators["atr"].to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        target = np.zeros(len(data), dtype=int)
        side = 0
        decision_bar = -1
        distance = 0.0
        stop = np.nan
        for i in range(len(data)):
            if side:
                if i == decision_bar + 1:
                    if np.isfinite(opens[i]) and opens[i] > 0:
                        # Anchor to the entry bar's open; entry ATR stays fixed.
                        stop = opens[i] - side * distance
                    else:
                        side = 0
                if side and (
                    not np.isfinite(closes[i])
                    or side * (closes[i] - stop) < 0
                    or i - decision_bar >= 4
                ):
                    side = 0
                target[i] = side
                # An exit always creates a flat bar before another entry.
                continue
            if (
                i >= self.warmup_bars(params) - 1
                and np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(score[i]) and abs(score[i]) >= 0.4
            ):
                side = 1 if score[i] > 0 else -1
                decision_bar = i
                distance = params.stop_atr * atr[i]
                stop = np.nan
                target[i] = side
        df = pd.DataFrame({"signal": target, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=40)
    prices = np.exp(4.5 + 0.002 * np.arange(40) ** 2)
    data = pd.DataFrame({
        "open": prices, "high": prices + 1.0, "low": prices - 1.0,
        "close": prices, "volume": 1000000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params()
    features = strategy.indicators(data, params)
    assert (features["topology"].iloc[params.graph_window - 1:] > 0.99).all()
    falling = data.copy()
    falling["close"] = np.exp(6.0 - 0.002 * np.arange(40) ** 2)
    assert (strategy.indicators(falling, params)["topology"].iloc[11:] < -0.99).all()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    features["topology"] = 1.0
    features["atr"] = 1.0
    data.loc[:, ["open", "close"]] = 100.0
    signals = strategy.generate_signals(data, features, ctx, params).data
    assert signals["signal"].iloc[14] == 0
    assert signals["signal"].iloc[15:19].eq(1).all()
    assert signals["signal"].iloc[19] == 0
    data.loc[index[16], "close"] = 97.0
    stopped = strategy.generate_signals(data, features, ctx, params).data
    assert stopped["signal"].iloc[16] == 1 and stopped["signal"].iloc[17] == 0
    features["topology"] = -1.0
    data.loc[index[16], "close"] = 103.0
    stopped = strategy.generate_signals(data, features, ctx, params).data
    assert stopped["signal"].iloc[16] == -1 and stopped["signal"].iloc[17] == 0
    prefix = data.iloc[:25]
    pd.testing.assert_frame_equal(
        strategy.indicators(prefix, params),
        strategy.indicators(data, params).iloc[:25],
    )


if __name__ == "__main__":
    _self_check()
