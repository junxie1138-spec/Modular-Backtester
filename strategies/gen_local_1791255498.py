from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAccelerationParams:
    percentile_window: int = 60
    tail_percentile: float = 0.90
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapAccelerationParams]):
    strategy_id = "gen_local_1791255498"

    @classmethod
    def params_type(cls) -> type[GapAccelerationParams]:
        return GapAccelerationParams

    def warmup_bars(self, params: GapAccelerationParams) -> int:
        if params.percentile_window < 2:
            raise ValueError("percentile_window must be at least 2")
        if not 0.5 < params.tail_percentile < 1.0:
            raise ValueError("tail_percentile must be between 0.5 and 1")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        return int(params.percentile_window + 3)

    def indicators(self, data: pd.DataFrame, params: GapAccelerationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda x: x > 0)
        out["gap_roc"] = data["open"].div(out["previous_close"]) - 1.0
        out["session_roc"] = data["close"].div(data["open"].where(data["open"] > 0)) - 1.0
        out["residual_gap"] = data["close"].div(out["previous_close"]) - 1.0
        out["gap_acceleration"] = out["gap_roc"].diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        history = out["gap_acceleration"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out["upper_threshold"] = history.quantile(params.tail_percentile)
        out["lower_threshold"] = history.quantile(1.0 - params.tail_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAccelerationParams,
    ) -> SignalFrame:
        valid = indicators.notna().all(axis=1)
        long_entry = (
            valid
            & (indicators["gap_roc"] < 0)
            & (indicators["gap_acceleration"] < 0)
            & (indicators["gap_acceleration"] < indicators["lower_threshold"])
            & (indicators["session_roc"] > 0)
            & (indicators["residual_gap"] < 0)
        )
        short_entry = (
            valid
            & (indicators["gap_roc"] > 0)
            & (indicators["gap_acceleration"] > 0)
            & (indicators["gap_acceleration"] > indicators["upper_threshold"])
            & (indicators["session_roc"] < 0)
            & (indicators["residual_gap"] > 0)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, candidate in enumerate(candidates):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                    continue
            elif candidate != 0:
                position = int(candidate)
                exit_bar = bar + params.hold_bars
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    direction = np.array([-1, 1, 1, 1, -1, -1, -1])
    derived = pd.DataFrame(
        {
            "gap_roc": direction * 0.02,
            "session_roc": direction * -0.005,
            "residual_gap": direction * 0.015,
            "gap_acceleration": direction * 0.02,
            "upper_threshold": 0.01,
            "lower_threshold": -0.01,
        },
        index=index,
    )
    for holding, expected in (
        (1, [0, 1, 0, -1, 0, 1, 0]),
        (2, [0, 1, 1, 0, -1, -1, 0]),
    ):
        result = GeneratedStrategy.generate_signals(
            None, data, derived, None, GapAccelerationParams(hold_bars=holding)
        ).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
