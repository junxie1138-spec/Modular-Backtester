from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    memory: int = 40
    atr_multiple: float = 2.5


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1791259026"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(params.memory) + 22

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if params.memory < 10 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("memory must be >= 10 and atr_multiple must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["latent_level"] = out["log_close"].rolling(5, min_periods=5).mean()
        out["slope"] = out["latent_level"].diff()
        out["baseline"] = out["slope"].rolling(10, min_periods=10).mean().shift(1)
        out["innovation"] = out["slope"] - out["baseline"]
        out["lag_one"] = out["innovation"].shift(1)
        out["lag_five"] = out["innovation"].shift(5)
        out["forecast_source"] = out["innovation"].shift(4)
        window = out["innovation"].rolling(params.memory, min_periods=params.memory)
        out["daily_correlation"] = window.corr(out["lag_one"])
        out["weekly_correlation"] = window.corr(out["lag_five"])
        out["weekly_covariance"] = window.cov(out["lag_five"])
        out["lag_variance"] = out["lag_five"].rolling(params.memory, min_periods=params.memory).var()
        out["echo_beta"] = out["weekly_covariance"] / out["lag_variance"].where(out["lag_variance"] > 0)
        out["forecast_slope"] = out["baseline"] + out["echo_beta"] * out["forecast_source"]
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            (out["weekly_correlation"] > 0.30)
            & (out["weekly_correlation"] > out["daily_correlation"] + 0.10)
            & (out["forecast_source"] > 0)
            & (out["forecast_slope"] > 0)
            & (out["slope"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        out["entry"] = out["setup"] & ~out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                held += 1
                # The first active iteration is the actual next-bar entry day.
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.atr_multiple * atr[i])
                if held >= 10 or (valid and close[i] <= stop):
                    active = False
                else:
                    signal[i] = 1
            elif valid and entries[i]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({"close": [100.0, 102.0, 104.0, 102.0] + [102.0] * 10}, index=index)
    indicators = pd.DataFrame({"entry": [True] + [False] * 13, "atr": [1.0, 1.0, 1.0, 10.0] + [10.0] * 10}, index=index)
    result = strategy.generate_signals(data, indicators, None, EchoParams(atr_multiple=1.0)).data
    assert result["signal"].tolist()[:5] == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    data["close"] = 100.0
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, None, EchoParams()).data
    assert result["signal"].tolist() == [0] + [1] * 10 + [0] * 3


if __name__ == "__main__":
    _self_check()
