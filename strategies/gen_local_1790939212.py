from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    topology_window: int = 16
    entropy_floor: float = 0.97
    breakout_window: int = 20
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18

    def __post_init__(self):
        if self.topology_window < 4 or self.breakout_window < 2 or self.atr_window < 2:
            raise ValueError("Lookbacks must be positive; topology_window must be at least four.")
        if not 0.0 < self.entropy_floor <= 1.0 or not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("Invalid entropy threshold or ATR multiplier.")
        if self.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1790939212"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(max(200, params.topology_window + 5, params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        spread = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["pressure"] = ((close - data["open"]) / spread).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["channel"] = data["high"].shift(1).rolling(params.breakout_window, min_periods=params.breakout_window).max()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        n = params.topology_window
        entropy = np.full(len(data), np.nan)
        if len(data) >= n:
            windows = np.lib.stride_tricks.sliding_window_view(out["pressure"].to_numpy(dtype=float), n)
            valid = np.isfinite(windows).all(axis=1)
            degrees = np.zeros(windows.shape, dtype=float)
            # ponytail: O(bars * window^2); batch windows if large histories exhaust memory.
            for left in range(n - 1):
                barrier = np.full(len(windows), -np.inf)
                for right in range(left + 1, n):
                    visible = (windows[:, left] > barrier) & (windows[:, right] > barrier)
                    degrees[:, left] += visible
                    degrees[:, right] += visible
                    barrier = np.maximum(barrier, windows[:, right])
            weights = degrees / np.maximum(degrees.sum(axis=1, keepdims=True), 1.0)
            terms = weights * np.log(np.maximum(weights, np.finfo(float).tiny))
            scores = -terms.sum(axis=1) / np.log(n)
            entropy[n - 1:] = np.where(valid, scores, np.nan)
        out["degree_entropy"] = entropy
        out["entropy_change"] = out["degree_entropy"] - out["degree_entropy"].shift(5)
        out["entry"] = (
            (close > out["ma200"])
            & (close > out["channel"])
            & (out["pressure"] > 0.0)
            & (out["degree_entropy"] >= params.entropy_floor)
            & (out["entropy_change"] > 0.0)
            & (out["atr"] > 0.0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TopologyParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    # The prior close requested entry; today's open is the fill proxy.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                stop_hit = np.isfinite(close[i]) and close[i] < stop
                if stop_hit or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    desired[i] = 1
            elif entries[i] and np.isfinite(close[i]) and close[i] > 0:
                desired[i] = 1
                pending = True
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = TopologyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000000.0}, index=index)
    indicators = strategy.indicators(data, p)
    assert indicators.index.equals(data.index)
    assert strategy.warmup_bars(p) >= 200
    indicators["entry"] = False
    indicators.loc[index[200], "entry"] = True
    indicators["atr"] = 2.0
    data.loc[index[201], "open"] = 110.0
    data.loc[index[201:203], "close"] = 109.0
    data.loc[index[203], "close"] = 104.0
    signals = strategy.generate_signals(data, indicators, None, p).data
    assert signals["signal"].iloc[200:205].tolist() == [0, 1, 1, 1, 0]
    data["close"] = 110.0
    signals = strategy.generate_signals(data, indicators, None, p).data
    assert signals["signal"].iloc[201:219].eq(1).all()
    assert signals["signal"].iloc[219] == 0
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()
