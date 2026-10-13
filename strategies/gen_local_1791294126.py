from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    correlation_window: int = 60
    volume_window: int = 20
    min_pairs: int = 8
    volume_threshold: float = 1.2
    correlation_floor: float = 0.15
    correlation_gap: float = 0.10
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791294126"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if not (
            params.correlation_window >= params.min_pairs >= 3
            and params.volume_window >= 2
            and params.hold_bars >= 1
            and np.isfinite(params.volume_threshold)
            and params.volume_threshold > 1.0
            and 0.0 < params.correlation_floor < 1.0
            and 0.0 < params.correlation_gap < 2.0
            and 0.0 < params.min_size <= params.max_size <= 1.0
        ):
            raise ValueError("Invalid strategy parameters")
        return int(params.volume_window + params.correlation_window + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        out["volume_baseline"] = data["volume"].rolling(params.volume_window).mean().shift(1)
        out["relative_volume"] = data["volume"].div(out["volume_baseline"].where(out["volume_baseline"] > 0))
        out["lag_relative_volume"] = out["relative_volume"].shift(1)
        out["high_volume_pair"] = out["lag_relative_volume"] >= params.volume_threshold
        out["ordinary_volume_pair"] = (
            out["lag_relative_volume"].notna()
            & (out["lag_relative_volume"] < params.volume_threshold)
        )
        for label, mask_column in (
            ("high", "high_volume_pair"),
            ("ordinary", "ordinary_volume_pair"),
        ):
            out[label + "_return"] = out["return"].where(out[mask_column])
            out[label + "_lag_return"] = out["lag_return"].where(out[mask_column])
            # Estimate the response to yesterday's volume regime; exclude today's pair.
            out[label + "_correlation"] = (
                out[label + "_return"]
                .rolling(params.correlation_window, min_periods=params.min_pairs)
                .corr(out[label + "_lag_return"])
                .shift(1)
                .replace([np.inf, -np.inf], np.nan)
                .clip(-1.0, 1.0)
            )
        out["correlation_spread"] = out["high_correlation"] - out["ordinary_correlation"]
        out["eligible"] = (
            out["high_correlation"].notna()
            & out["ordinary_correlation"].notna()
            & out["return"].notna()
            & (out["relative_volume"] >= params.volume_threshold)
            & (out["high_correlation"].abs() >= params.correlation_floor)
            & (out["correlation_spread"].abs() >= params.correlation_gap)
            & (out["high_correlation"] * out["correlation_spread"] > 0.0)
        )
        out["entry_direction"] = np.sign(
            out["return"] * out["high_correlation"]
        ).where(out["eligible"], 0).fillna(0).astype(int)
        out["strength"] = (
            out["high_correlation"].abs()
            * out["relative_volume"].div(params.volume_threshold)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        directions = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        position = 0
        exit_at = -1
        entry_size = params.min_size
        for i in range(warmup - 1, len(data)):
            if position:
                if i == exit_at:
                    position = 0
                    continue
            elif directions[i]:
                position = int(directions[i])
                exit_at = i + params.hold_bars
                entry_size = params.min_size + (
                    params.max_size - params.min_size
                ) * strengths[i]
            signal[i] = position
            if position:
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    first = strategy.warmup_bars(params) - 1
    index = pd.date_range("2020-01-01", periods=first + 2 * params.hold_bars + 5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"entry_direction": -1, "strength": 0.9}, index=index
    )
    indicators.iloc[first] = [1, 0.3]
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:first + 1].eq(0).all()
    assert result["signal"].iloc[first + 1:first + 1 + params.hold_bars].eq(1).all()
    assert result["signal"].iloc[first + params.hold_bars + 1] == 0
    second = first + params.hold_bars + 2
    assert result["signal"].iloc[second:second + params.hold_bars].eq(-1).all()
    assert result["signal"].iloc[second + params.hold_bars] == 0
    assert result["size"].iloc[first + 1:first + 1 + params.hold_bars].nunique() == 1
    assert result["size"].iloc[second] > result["size"].iloc[first + 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
