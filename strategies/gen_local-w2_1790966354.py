from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExcursionParams:
    path_bars: int = 12
    calibration_bars: int = 126
    area_percentile: float = 0.80
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ExcursionParams]):
    strategy_id = "gen_local-w2_1790966354"

    @classmethod
    def params_type(cls) -> type[ExcursionParams]:
        return ExcursionParams

    def warmup_bars(self, params: ExcursionParams) -> int:
        if params.path_bars < 2 or params.calibration_bars < 2 or params.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two and holding period positive")
        if not 0.0 < params.area_percentile < 1.0:
            raise ValueError("area_percentile must lie strictly between zero and one")
        return int(params.path_bars + params.calibration_bars + 1)

    def indicators(self, data: pd.DataFrame, params: ExcursionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        length = params.path_bars
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["up_excursion"] = ((data["high"] - previous_close) / previous_close).clip(lower=0)
        out["down_excursion"] = ((previous_close - data["low"]) / previous_close).clip(lower=0)
        up = out["up_excursion"].replace([np.inf, -np.inf], np.nan)
        down = out["down_excursion"].replace([np.inf, -np.inf], np.nan)
        out["up_excursion"] = up
        out["down_excursion"] = down
        out["up_total"] = up.rolling(length, min_periods=length).sum()
        out["down_total"] = down.rolling(length, min_periods=length).sum()
        area = pd.Series(0.0, index=data.index)
        # Second-order antisymmetric signature: sum over all ordered pairs.
        # ponytail: O(path_bars * rows); use prefix algebra if windows become large.
        for lag in range(1, length):
            cross = up.shift(lag) * down - down.shift(lag) * up
            area = area + cross.rolling(length - lag, min_periods=length - lag).sum()
        out["signed_area"] = area
        total = out["up_total"] + out["down_total"]
        out["area"] = (area / total.pow(2).where(total > 0)).replace([np.inf, -np.inf], np.nan)
        out["area_threshold"] = out["area"].abs().rolling(
            params.calibration_bars, min_periods=params.calibration_bars
        ).quantile(params.area_percentile).shift(1)
        previous_high = data["high"].shift(1)
        previous_low = data["low"].shift(1)
        intersection = (np.minimum(data["high"], previous_high) - np.maximum(data["low"], previous_low)).clip(lower=0)
        union = np.maximum(data["high"], previous_high) - np.minimum(data["low"], previous_low)
        out["range_overlap"] = (intersection / union.where(union > 0)).replace([np.inf, -np.inf], np.nan)
        out["overlap_mean"] = out["range_overlap"].rolling(length, min_periods=length).mean()
        out["overlap_threshold"] = out["overlap_mean"].rolling(
            params.calibration_bars, min_periods=params.calibration_bars
        ).quantile(0.5).shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExcursionParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        needed = indicators[["area", "area_threshold", "overlap_mean", "overlap_threshold"]]
        ready = np.isfinite(needed).all(axis=1)
        extreme = ready & (indicators["area"].abs() > indicators["area_threshold"])
        directional = indicators["overlap_mean"] <= indicators["overlap_threshold"]
        rotation = np.sign(indicators["area"].fillna(0))
        proposals = np.where(extreme, np.where(directional, -rotation, rotation), 0).astype(int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar >= params.hold_bars:
                    position = 0
            elif proposals[i] != 0:
                position = int(proposals[i])
                entry_bar = i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ExcursionParams()
    index = pd.date_range("2020-01-01", periods=2 * params.hold_bars + 5, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "area": 1.0,
        "area_threshold": 0.5,
        "overlap_mean": 0.1,
        "overlap_threshold": 0.5,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[0] == 0
    assert (result["signal"].iloc[1:params.hold_bars + 1] == -1).all()
    assert result["signal"].iloc[params.hold_bars + 1] == 0
    assert result["signal"].iloc[params.hold_bars + 2] == -1
    indicators.loc[:, "area_threshold"] = np.nan
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert (result["signal"] == 0).all()
