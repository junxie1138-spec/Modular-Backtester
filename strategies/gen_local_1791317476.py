from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 20
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791317476"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.trend_window < 5 or params.hold_bars < 1:
            raise ValueError("trend_window must be >= 5 and hold_bars >= 1")
        return int(params.trend_window + 10)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        w = params.trend_window
        out = pd.DataFrame(index=data.index)
        midpoint = (data["high"] + data["low"]) / 2.0
        out["mid_log"] = np.log(midpoint.where(midpoint > 0)).replace([np.inf, -np.inf], np.nan)
        out["close_log"] = np.log(data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["return"] = out["close_log"].diff()
        out["volatility"] = out["return"].rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
        weights = np.arange(w, dtype=float) - (w - 1) / 2.0
        weights /= np.dot(weights, weights)

        for source, target in (("mid_log", "mid_slope"), ("close_log", "close_slope")):
            values = out[source]
            if len(values) == 0:
                out[target] = pd.Series(index=data.index, dtype=float)
                continue
            slope = np.convolve(values.fillna(0.0).to_numpy(), weights[::-1], mode="full")[:len(values)]
            valid = values.rolling(w, min_periods=w).count().eq(w)
            out[target] = pd.Series(slope, index=data.index).where(valid)

        out["mid_innovation"] = out["mid_slope"].diff(5) / out["volatility"]
        out["close_innovation"] = out["close_slope"].diff(5) / out["volatility"]
        out["mid_lead"] = out["mid_innovation"].shift(5) - out["close_innovation"].shift(5)
        out["entry"] = (
            out["mid_innovation"].shift(5).gt(0.08)
            & out["mid_lead"].gt(0.08)
            & out["close_innovation"].gt(0.08)
            & out["close_innovation"].shift(1).le(0.08)
            & out["close_slope"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, entry in enumerate(entries):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                    continue
                signal[i] = 1
            elif entry:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params(hold_bars=18)
    index = pd.date_range("2020-01-01", periods=42, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals["signal"].tolist() == [0] + [1] * 18 + [0] + [1] * 18 + [0] + [1] * 3
    flat = strategy.indicators(data, params)
    assert not flat["entry"].any()
    assert strategy.warmup_bars(params) >= params.trend_window + 10
