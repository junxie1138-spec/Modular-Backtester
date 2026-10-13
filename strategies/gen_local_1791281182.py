from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangeParams:
    reference_window: int = 40
    jump_z: float = 2.0


class GeneratedStrategy(BaseStrategy[ChangeParams]):
    strategy_id = "gen_local_1791281182"

    @classmethod
    def params_type(cls) -> type[ChangeParams]:
        return ChangeParams

    @staticmethod
    def warmup_bars(params: ChangeParams) -> int:
        return 32 + max(8, int(params.reference_window)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: ChangeParams) -> pd.DataFrame:
        window = max(8, int(params.reference_window))
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        weights = np.empty(32, dtype=float)
        weights[0] = 1.0
        for k in range(1, 32):
            weights[k] = -weights[k - 1] * (0.45 - k + 1) / k
        fractional_close = close * weights[0]
        for k in range(1, 32):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        old = fractional_close.shift(4)
        baseline = old.rolling(window, min_periods=window).mean()
        scale = old.rolling(window, min_periods=window).std(ddof=1)
        recent = fractional_close.rolling(4, min_periods=4).mean()
        denominator = scale.where(scale > 0) * np.sqrt(0.25 + 1.0 / window)
        jump = ((recent - baseline) / denominator).replace([np.inf, -np.inf], np.nan)
        old_drift = old - fractional_close.shift(window + 4)
        entry = (
            (jump > params.jump_z)
            & (jump.shift(1) <= params.jump_z)
            & (old_drift < 0)
        ).fillna(False)
        return pd.DataFrame(
            {
                "fractional_close": fractional_close,
                "baseline": baseline,
                "baseline_scale": scale,
                "recent_mean": recent,
                "jump_z": jump,
                "old_drift": old_drift,
                "entry": entry,
            },
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangeParams,
    ) -> SignalFrame:
        entries = indicators["entry"].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, enter in enumerate(entries):
            if remaining > 0:
                remaining -= 1
            elif enter:
                remaining = 4
            signal[i] = int(remaining > 0)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=11, freq="D")
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, ChangeParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    flat = GeneratedStrategy.indicators(data, ChangeParams())
    assert not flat["entry"].any()


if __name__ == "__main__":
    _self_check()
