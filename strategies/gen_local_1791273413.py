from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 16
    coherence_window: int = 7
    percentile_window: int = 126
    entry_percentile: float = 0.85
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791273413"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_window + 1, params.slope_window + params.coherence_window + params.percentile_window + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_return"] = out["log_close"].diff()
        out["slope"] = out["log_return"].rolling(params.slope_window).mean()
        out["predicted_slope"] = out["slope"].shift(1)
        out["innovation"] = out["log_return"] - out["predicted_slope"]
        out["innovation_sum"] = out["innovation"].rolling(params.coherence_window).sum()
        out["absolute_innovation_sum"] = out["innovation"].abs().rolling(params.coherence_window).sum()
        out["coherence"] = out["innovation_sum"] / out["absolute_innovation_sum"].replace(0.0, np.nan)
        out["threshold"] = out["coherence"].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = pd.Series(np.isfinite(indicators[["slope", "innovation", "coherence", "threshold", "atr"]]).all(axis=1), index=data.index)
        setup = (ready & (indicators["slope"] > 0.0)
                 & (indicators["innovation"] > 0.0)
                 & (indicators["coherence"] > 0.0)
                 & (indicators["coherence"] > indicators["threshold"])
                 & (indicators["atr"] > 0.0)).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if active and i == entry_bar:
                # The prior decision fills at this open; freeze its ATR distance.
                if not np.isfinite(opening[i]) or opening[i] <= 0.0:
                    active = False
                else:
                    stop = opening[i] - params.stop_atr * atr[i - 1]
            if active:
                if not np.isfinite(close[i]) or close[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
            elif setup[i]:
                active = True
                entry_bar = i + 1
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # One check covers delayed entry, next-open anchoring, and a nontrailing stop.
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0, 110.0, 115.0, 110.0, 107.0, 101.0, 101.0],
        "high": [101.0, 116.0, 116.0, 111.0, 108.0, 102.0, 102.0],
        "low": [99.0, 109.0, 109.0, 106.0, 100.0, 100.0, 100.0],
        "close": [100.0, 115.0, 110.0, 107.0, 101.0, 101.0, 101.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "slope": 0.01, "innovation": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "coherence": 0.9, "threshold": 0.8, "atr": 1.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(stop_atr=2.0)
    result = strategy.generate_signals(data, ind, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    params.hold_bars = 2
    result = strategy.generate_signals(data, ind, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
