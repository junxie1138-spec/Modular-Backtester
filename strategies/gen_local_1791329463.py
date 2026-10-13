from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    pressure_bars: int = 3
    rank_bars: int = 6
    rank_jump: float = 0.5
    hold_bars: int = 4

    def __post_init__(self):
        if self.pressure_bars < 2 or self.rank_bars < 2:
            raise ValueError("Windows must be at least two bars.")
        if self.pressure_bars + self.rank_bars > 10:
            raise ValueError("Combined warmup must not exceed ten bars.")
        if not 0.0 < self.rank_jump <= 1.0 or not 3 <= self.hold_bars <= 5:
            raise ValueError("Invalid rank jump or holding period.")


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local_1791329463"

    @classmethod
    def params_type(cls):
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        # Nested windows need p + r - 1 bars; the rank change needs one more.
        return int(params.pressure_bars + params.rank_bars)

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        out["location"] = (
            (2.0 * data["close"] - data["high"] - data["low"]) / out["range"]
        ).clip(-1.0, 1.0).replace([np.inf, -np.inf], np.nan)
        out["valid_volume"] = data["volume"].where(
            np.isfinite(data["volume"]) & (data["volume"] > 0)
            & out["location"].notna()
        )
        out["signed_volume"] = out["location"] * out["valid_volume"]
        out["pressure_sum"] = out["signed_volume"].rolling(
            params.pressure_bars, min_periods=params.pressure_bars
        ).sum()
        out["volume_sum"] = out["valid_volume"].rolling(
            params.pressure_bars, min_periods=params.pressure_bars
        ).sum()
        out["pressure"] = out["pressure_sum"] / out["volume_sum"].where(
            out["volume_sum"] > 0
        )
        out["rank"] = out["pressure"].rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).rank(method="average", pct=True)
        out["rank_change"] = out["rank"].diff()
        out["previous_pressure"] = out["pressure"].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PressureParams,
    ) -> SignalFrame:
        up = (
            (indicators["rank_change"] >= params.rank_jump)
            & (indicators["rank"] >= 0.8)
            & (indicators["previous_pressure"] <= 0)
            & (indicators["pressure"] > 0)
        )
        down = (
            (indicators["rank_change"] <= -params.rank_jump)
            & (indicators["rank"] <= 0.2)
            & (indicators["previous_pressure"] >= 0)
            & (indicators["pressure"] < 0)
        )
        candidates = np.select(
            [up.fillna(False), down.fillna(False)], [1, -1], default=0
        )
        signal = np.zeros(len(data), dtype=int)
        side = 0
        exit_at = -1
        for i, candidate in enumerate(candidates):
            if side != 0:
                if i == exit_at:
                    side = 0
            elif candidate != 0:
                side = int(candidate)
                exit_at = i + params.hold_bars
            signal[i] = side
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = PressureParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0},
        index=index,
    )
    ind = pd.DataFrame(
        {
            "pressure": [1.0] * 6 + [-1.0] * 6,
            "previous_pressure": [-1.0] * 6 + [1.0] * 6,
            "rank": [1.0] * 6 + [0.1] * 6,
            "rank_change": [1.0] * 6 + [-1.0] * 6,
        },
        index=index,
    )
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, -1]
    assert (result["size"] > 0).all()
    actual = GeneratedStrategy.indicators(None, data, params)
    assert actual["rank_change"].iloc[:8].isna().all()
    flat = GeneratedStrategy.generate_signals(None, data, actual, None, params).data
    assert (flat["signal"] == 0).all()
    assert GeneratedStrategy.warmup_bars(None, params) == 9


if __name__ == "__main__":
    _self_check()
