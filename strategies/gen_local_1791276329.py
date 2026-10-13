from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolumeParams:
    seasonal_lookback: int = 126
    volume_multiple: float = 1.15
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalVolumeParams]):
    strategy_id = "gen_local_1791276329"

    @classmethod
    def params_type(cls):
        return SeasonalVolumeParams

    @staticmethod
    def warmup_bars(params: SeasonalVolumeParams) -> int:
        if params.seasonal_lookback < 40:
            raise ValueError("seasonal_lookback must be at least 40")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive")
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be 3, 4, or 5")
        return int(params.seasonal_lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalVolumeParams) -> pd.DataFrame:
        warmup = GeneratedStrategy.warmup_bars(params)
        window = params.seasonal_lookback
        out = pd.DataFrame(index=data.index)
        out["weekday"] = data.index.dayofweek
        volume = data["volume"].where(
            np.isfinite(data["volume"]) & data["volume"].gt(0)
        )
        baseline = pd.Series(0.0, index=data.index)
        for weekday in range(5):
            selected = out["weekday"].eq(weekday) & volume.notna()
            total = volume.where(selected, 0.0).shift(1).rolling(
                window, min_periods=window
            ).sum()
            count = selected.astype(float).shift(1).rolling(
                window, min_periods=window
            ).sum()
            average = total.div(count.where(count.ge(4)))
            baseline = baseline.where(out["weekday"].ne(weekday), average)
        out["weekday_volume_baseline"] = baseline.fillna(0.0)
        out["relative_volume"] = volume.div(baseline.where(baseline.gt(0))).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        out["daily_return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        out["positive_body"] = data["close"].gt(data["open"])
        out["confirmed_move"] = (
            out["daily_return"].gt(0)
            & out["positive_body"]
            & out["relative_volume"].ge(params.volume_multiple)
        )
        out["two_bar_confirmation"] = (
            out["confirmed_move"]
            & out["confirmed_move"].shift(1, fill_value=False)
        )
        out["entry_ready"] = (
            out["two_bar_confirmation"]
            & out["weekday"].eq(4)
            & out["weekday"].shift(1).eq(3)
            & (np.arange(len(data)) >= warmup - 1)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalVolumeParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar < exit_bar:
                    signal[bar] = 1
                else:
                    exit_bar = -1
                continue
            if entries[bar]:
                signal[bar] = 1
                exit_bar = bar + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        # Both entry and exit decisions execute on the following bar.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2020-01-02", periods=12)
    sample = pd.DataFrame(index=index)
    evidence = pd.DataFrame({"entry_ready": False}, index=index)
    evidence.iloc[[1, 2, 6], 0] = True
    result = GeneratedStrategy.generate_signals(
        sample, evidence, None, SeasonalVolumeParams()
    ).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    assert result["size"].gt(0).all()
