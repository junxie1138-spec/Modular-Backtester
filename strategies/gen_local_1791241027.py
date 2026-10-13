from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    rank_window: int = 60
    atr_window: int = 14
    gap_percentile: float = 0.90
    volume_percentile: float = 0.25
    trail_atr: float = 0.80
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1791241027"

    @classmethod
    def params_type(cls) -> type[GapParams]:
        return GapParams

    @staticmethod
    def warmup_bars(params: GapParams) -> int:
        return int(max(params.rank_window, params.atr_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        if not (
            params.rank_window >= 2
            and params.atr_window >= 1
            and 0.0 < params.gap_percentile <= 1.0
            and 0.0 < params.volume_percentile <= 1.0
            and params.trail_atr > 0.0
            and params.hold_bars in (1, 2)
        ):
            raise ValueError("Invalid gap strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        previous = out["previous_close"].where(out["previous_close"] > 0)
        out["gap"] = data["open"].div(previous).sub(1.0)
        gap_points = data["open"] - previous
        out["retained_gap"] = (data["close"] - previous).div(
            gap_points.where(gap_points > 0)
        )
        out["gap_rank"] = out["gap"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        volume = data["volume"].where(data["volume"] > 0)
        out["volume_rank"] = volume.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous).abs(),
                (data["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["gap_primitive"] = (
            (out["gap"] > 0)
            & (out["gap_rank"] >= params.gap_percentile)
            & (out["retained_gap"] >= 0.5)
            & (out["retained_gap"] < 1.0)
        ).fillna(False)
        out["volume_primitive"] = (
            out["volume_rank"] <= params.volume_percentile
        ).fillna(False)
        out["entry"] = (
            out["gap_primitive"]
            & out["volume_primitive"]
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0
            valid = valid and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if valid:
                    # First held close occurs one bar after the entry decision.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or closes[i] <= stop or age >= params.hold_bars:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
                    # Exit takes priority over a fresh entry on this close.
                    continue
                raw[i] = 1
            elif valid and entries[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
                raw[i] = 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 101., 100., 99., 102., 103.]}, index=index)
    ind = pd.DataFrame(
        {"atr": [1., 1., 1., 1., 1., 1.],
         "entry": [True, True, True, False, False, False]},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, ind, None, GapParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    ind.loc[index[1], "atr"] = np.nan
    result = GeneratedStrategy.generate_signals(data, ind, None, GapParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
