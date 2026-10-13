from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    range_bars: int = 6
    volume_multiple: float = 1.3
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1791332913"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        if not 3 <= params.range_bars <= 9:
            raise ValueError("range_bars must be between 3 and 9")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive")
        return int(max(params.range_bars + 1, 5))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["change"] = data["close"].diff()
        for lag in (1, 2, 3):
            out[f"change_lag{lag}"] = out["change"].shift(lag)
            out[f"volume_lag{lag}"] = data["volume"].shift(lag)
        out["ceiling"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["floor"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out["volume_base"] = data["volume"].rolling(params.range_bars, min_periods=params.range_bars).mean().shift(1)
        out["volume_ratio"] = data["volume"].div(out["volume_base"].where(out["volume_base"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["alternating"] = ((out["change_lag1"] * out["change_lag2"] < 0) & (out["change_lag2"] * out["change_lag3"] < 0))
        out["volume_ebb"] = ((out["volume_lag1"] > 0) & (out["volume_lag1"] < out["volume_lag2"]) & (out["volume_lag2"] < out["volume_lag3"]))
        ready = out["alternating"] & out["volume_ebb"] & out["volume_ratio"].ge(params.volume_multiple)
        out["setup"] = np.select([ready & data["close"].gt(out["ceiling"]), ready & data["close"].lt(out["floor"])], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        self.warmup_bars(params)
        setup = indicators["setup"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(setup):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signal[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"setup": [1] + [-1] * 9 + [1, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(WaveParams(range_bars=9)) == 10


if __name__ == "__main__":
    _self_check()
