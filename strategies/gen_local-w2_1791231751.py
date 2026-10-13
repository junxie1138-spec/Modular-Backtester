from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletSeasonParams:
    history_bars: int = 130
    energy_smoothing: int = 4
    entry_score: float = 0.12
    full_size_score: float = 0.45
    min_size: float = 0.15


class GeneratedStrategy(BaseStrategy[WaveletSeasonParams]):
    strategy_id = "gen_local-w2_1791231751"

    @classmethod
    def params_type(cls):
        return WaveletSeasonParams

    def warmup_bars(self, params: WaveletSeasonParams) -> int:
        return int(params.history_bars + params.energy_smoothing + 12)

    def indicators(self, data: pd.DataFrame, params: WaveletSeasonParams) -> pd.DataFrame:
        if params.history_bars < 60 or params.energy_smoothing < 1:
            raise ValueError("history_bars must be >= 60; energy_smoothing must be >= 1")
        if not 0 < params.entry_score <= params.full_size_score or not 0 < params.min_size <= 1:
            raise ValueError("Invalid score thresholds or minimum size")
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff().replace(
            [np.inf, -np.inf], np.nan
        )
        for scale in (2, 4, 8):
            half = scale // 2
            recent = out["return"].rolling(half, min_periods=half).sum()
            out[f"haar_{scale}"] = (recent - recent.shift(half)) / np.sqrt(scale)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_smoothing, min_periods=params.energy_smoothing
            ).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(
            axis=1, min_count=3
        )
        out["fine_share"] = (
            (out["energy_2"] + out["energy_4"]) / out["total_energy"].replace(0, np.nan)
        ).fillna(0.5).clip(0.05, 0.95).where(out["total_energy"].notna())
        # Outcomes mature four sessions after their originating decision bar.
        out["matured_return"] = out["return"].rolling(4, min_periods=4).sum()
        out["origin_fine_share"] = out["fine_share"].shift(4)
        out["weekday"] = pd.Series(data.index.dayofweek, index=data.index)
        out["origin_weekday"] = out["weekday"].shift(4)
        forecast = pd.Series(np.nan, index=data.index, dtype=float)
        risk = pd.Series(np.nan, index=data.index, dtype=float)
        valid = out["matured_return"].notna() & out["origin_fine_share"].notna()
        for weekday in range(5):
            mask = valid & out["origin_weekday"].eq(weekday)
            out[f"count_{weekday}"] = mask.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            for name, weight in (
                ("fine", out["origin_fine_share"]),
                ("coarse", 1.0 - out["origin_fine_share"]),
            ):
                out[f"{name}_mass_{weekday}"] = weight.where(mask, 0.0).rolling(
                    params.history_bars, min_periods=params.history_bars
                ).sum()
                out[f"{name}_sum_{weekday}"] = (
                    out["matured_return"] * weight
                ).where(mask, 0.0).rolling(
                    params.history_bars, min_periods=params.history_bars
                ).sum()
                out[f"{name}_mean_{weekday}"] = (
                    out[f"{name}_sum_{weekday}"] /
                    out[f"{name}_mass_{weekday}"].replace(0, np.nan)
                ).where(out[f"count_{weekday}"] >= 10)
            out[f"rms_{weekday}"] = np.sqrt(
                out["matured_return"].pow(2).where(mask, 0.0).rolling(
                    params.history_bars, min_periods=params.history_bars
                ).sum() / out[f"count_{weekday}"].replace(0, np.nan)
            )
            today = out["weekday"].eq(weekday)
            estimate = (
                out["fine_share"] * out[f"fine_mean_{weekday}"] +
                (1.0 - out["fine_share"]) * out[f"coarse_mean_{weekday}"]
            )
            forecast = forecast.where(~today, estimate)
            risk = risk.where(~today, out[f"rms_{weekday}"])
        out["forecast"] = forecast
        out["risk"] = risk
        out["score"] = (forecast / risk.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).clip(-1.0, 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletSeasonParams,
    ) -> SignalFrame:
        score = indicators["score"].reindex(data.index).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        desired = np.where(
            score >= params.entry_score, 1,
            np.where(score <= -params.entry_score, -1, 0),
        )
        positions = np.zeros(len(data), dtype=int)
        state = 0
        for i, direction in enumerate(desired):
            if state == 0:
                state = int(direction)
            elif direction == -state:
                # Opposite qualified entry closes the position before re-entry.
                state = 0
            positions[i] = state
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["size"] = (score.abs() / params.full_size_score).clip(
            lower=params.min_size, upper=1.0
        ).shift(1).fillna(params.min_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveletSeasonParams()
    index = pd.bdate_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": np.ones(6)}, index=index)
    indicators = pd.DataFrame({"score": [0.3, 0.0, -0.3, -0.3, 0.3, 0.3]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert (result["size"] > 0).all()
    assert np.isclose(result["size"].iloc[1], 0.3 / params.full_size_score)
    index = pd.bdate_range("2020-01-01", periods=360)
    close = 100.0 * np.exp(np.cumsum(0.002 * np.sin(np.arange(360) * 0.6)))
    data = pd.DataFrame({"close": close}, index=index)
    original = strategy.indicators(data, params)
    changed = data.copy()
    changed.iloc[-1, 0] *= 1.2
    pd.testing.assert_frame_equal(original.iloc[:-1], strategy.indicators(changed, params).iloc[:-1])


if __name__ == "__main__":
    _self_check()
