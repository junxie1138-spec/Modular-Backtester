from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExcursionParams:
    ma_window: int = 20
    percentile_window: int = 126
    tail_quantile: float = 0.15
    atr_window: int = 14
    trail_atr: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[ExcursionParams]):
    strategy_id = "gen_local-w2_1791303105"

    @classmethod
    def params_type(cls) -> type[ExcursionParams]:
        return ExcursionParams

    def warmup_bars(self, params: ExcursionParams) -> int:
        return int(max(params.ma_window + params.percentile_window + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ExcursionParams) -> pd.DataFrame:
        if (params.ma_window < 2 or params.percentile_window < 3
                or params.atr_window < 1 or params.holding_bars < 1
                or not 0.0 < params.tail_quantile < 0.5
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["lower"] = out["z"].rolling(params.percentile_window).quantile(params.tail_quantile).shift(1)
        out["upper"] = out["z"].rolling(params.percentile_window).quantile(1.0 - params.tail_quantile).shift(1)
        out["previous_close"] = close.shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["below"] = out["z"] < out["lower"]
        out["above"] = out["z"] > out["upper"]
        out["long_setup"] = (
            out["below"].shift(1, fill_value=False)
            & out["below"].shift(2, fill_value=False)
            & (out["z"] >= out["lower"])
            & (out["z"] < 0.0)
        )
        out["short_setup"] = (
            out["above"].shift(1, fill_value=False)
            & out["above"].shift(2, fill_value=False)
            & (out["z"] <= out["upper"])
            & (out["z"] > 0.0)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ExcursionParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        extreme = 0.0
        stop = 0.0
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                position = 0
                continue
            if position:
                if position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                    stopped = close[i] <= stop
                else:
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                    stopped = close[i] >= stop
                if stopped or i - entry_bar >= params.holding_bars:
                    position = 0
                    continue
            else:
                if longs[i]:
                    position = 1
                elif shorts[i]:
                    position = -1
                if position:
                    entry_bar = i
                    extreme = close[i]
                    stop = close[i] - position * params.trail_atr * atr[i]
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ExcursionParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 101., 98., 98., 96., 99., 100.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.] * 7,
        "long_setup": [True, False, False, False, False, False, False],
        "short_setup": [False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    indicators["short_setup"] = False
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
