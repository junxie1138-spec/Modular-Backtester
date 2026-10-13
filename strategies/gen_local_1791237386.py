from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInventoryParams:
    memory: int = 4
    volume_window: int = 20
    volume_confirm: float = 1.25
    min_gap: float = 0.0015
    threshold: float = 0.45


class GeneratedStrategy(BaseStrategy[GapInventoryParams]):
    strategy_id = "gen_local_1791237386"

    @classmethod
    def params_type(cls) -> type[GapInventoryParams]:
        return GapInventoryParams

    def warmup_bars(self, params: GapInventoryParams) -> int:
        if params.memory < 1 or params.volume_window < 1:
            raise ValueError("Lookback windows must be positive.")
        if params.volume_confirm <= 0 or params.min_gap <= 0:
            raise ValueError("Volume confirmation and minimum gap must be positive.")
        if not 0 < params.threshold <= 1:
            raise ValueError("Threshold must lie in (0, 1].")
        return int(params.volume_window + params.memory)

    def indicators(self, data: pd.DataFrame, params: GapInventoryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan).where(data["close"] > 0)
        opening = data["open"].replace([np.inf, -np.inf], np.nan).where(data["open"] > 0)
        volume = data["volume"].replace([np.inf, -np.inf], np.nan).where(data["volume"] >= 0)
        out["previous_close"] = close.shift(1)
        out["gap"] = opening - out["previous_close"]
        out["gap_fraction"] = out["gap"] / out["previous_close"]
        out["day_move"] = close - opening
        out["direction"] = np.sign(out["gap"])
        out["volume_baseline"] = volume.shift(1).rolling(params.volume_window, min_periods=params.volume_window).mean()
        out["relative_volume"] = volume / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["confirmed"] = (
            (out["gap_fraction"].abs() >= params.min_gap)
            & (out["relative_volume"] >= params.volume_confirm)
            & (out["direction"] * out["day_move"] > 0)
        )
        out["cohort_weight"] = (
            out["gap_fraction"].abs()
            * out["relative_volume"].clip(upper=3.0)
        ).where(out["confirmed"], 0.0).fillna(0.0)
        out["surviving_inventory"] = 0.0
        out["original_inventory"] = 0.0
        for lag in range(params.memory):
            gap = out["gap"].shift(lag)
            direction = out["direction"].shift(lag)
            anchor = out["previous_close"].shift(lag)
            weight = out["cohort_weight"].shift(lag).fillna(0.0)
            retention = (
                direction * (close - anchor) / gap.abs().where(gap.abs() > 0)
            ).clip(lower=0.0, upper=1.0).fillna(0.0)
            out[f"retention_{lag}"] = retention
            out[f"inventory_{lag}"] = (direction * retention * weight).fillna(0.0)
            out["surviving_inventory"] += out[f"inventory_{lag}"]
            out["original_inventory"] += weight
        out["ready"] = (
            out["volume_baseline"].shift(params.memory - 1).notna()
            & close.notna()
            & opening.notna()
            & volume.notna()
        )
        out["score"] = (
            out["surviving_inventory"]
            / out["original_inventory"].where(out["original_inventory"] > 0)
        ).where(out["ready"], 0.0).fillna(0.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapInventoryParams,
    ) -> SignalFrame:
        score = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ready = indicators["ready"].fillna(False).astype(bool)
        votes = np.where(ready & (score >= params.threshold), 1,
                         np.where(ready & (score <= -params.threshold), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, vote in enumerate(votes):
            if position == 0:
                position = int(vote)
            elif vote == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "score": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
        "ready": [True] * 8,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, GapInventoryParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
