from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    shock_z: float = 1.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791339211"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.shock_z) or params.shock_z <= 0:
            raise ValueError("shock_z must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        out["impulse"] = out["return"].shift(2)
        # The baseline ends before the impulse and both unloading sessions.
        out["baseline_vol"] = out["return"].shift(3).rolling(20, min_periods=20).std(ddof=0)
        out["retained_displacement"] = (
            (1.0 + out["impulse"])
            * (1.0 + out["previous_return"])
            * (1.0 + out["return"])
            - 1.0
        )
        out["entry"] = (
            out[["return", "previous_return", "impulse", "baseline_vol", "retained_displacement"]].notna().all(axis=1)
            & (out["baseline_vol"] > 0)
            & (out["impulse"] > params.shock_z * out["baseline_vol"])
            & (out["previous_return"] < 0)
            & (out["return"] < 0)
            & (out["retained_displacement"] >= 0.65 * out["impulse"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, int) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, trigger in enumerate(entry):
            if exit_at >= 0:
                if i == exit_at:
                    # Force a flat bar even if another entry qualifies here.
                    exit_at = -1
                else:
                    signal[i] = 1
            elif trigger:
                exit_at = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.iloc[[0, 8, 9], 0] = True
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] + [1] * 8 + [0] * 7
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
