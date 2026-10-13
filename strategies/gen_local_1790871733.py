from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    trend_window: int = 20
    noise_window: int = 30
    queue_window: int = 3
    overflow_threshold: float = 1.2
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local_1790871733"

    @classmethod
    def params_type(cls) -> type[InnovationParams]:
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        if min(params.trend_window, params.noise_window, params.queue_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        if not np.isfinite(params.overflow_threshold) or params.overflow_threshold <= 0:
            raise ValueError("overflow_threshold must be finite and positive")
        return int(params.trend_window + params.noise_window + params.queue_window + 3)

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        out["log_return"] = out["log_close"].diff()
        out["latent_slope"] = out["log_return"].rolling(
            params.trend_window, min_periods=params.trend_window
        ).mean()
        out["predicted_slope"] = out["latent_slope"].shift(1)
        out["innovation"] = out["log_return"] - out["predicted_slope"]
        out["innovation_scale"] = out["innovation"].rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).shift(1).replace(0.0, np.nan)
        out["standardized_innovation"] = (
            out["innovation"] / out["innovation_scale"]
        ).replace([np.inf, -np.inf], np.nan)
        out["evidence"] = out["standardized_innovation"].rolling(
            params.queue_window, min_periods=params.queue_window
        ).sum() / np.sqrt(params.queue_window)
        out["up_overflow"] = (
            out["evidence"].gt(params.overflow_threshold)
            & out["standardized_innovation"].gt(0)
        )
        out["down_overflow"] = (
            out["evidence"].lt(-params.overflow_threshold)
            & out["standardized_innovation"].lt(0)
        )
        out["long_confirmed"] = (
            out["up_overflow"] & out["up_overflow"].shift(1, fill_value=False)
        )
        out["short_confirmed"] = (
            out["down_overflow"] & out["down_overflow"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InnovationParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        candidates = np.where(
            indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool),
            1,
            np.where(
                indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool),
                -1,
                0,
            ),
        )
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0 and i == exit_bar:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = InnovationParams()
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    candidates = np.array([0, 1, 0, -1, 0, 0, -1, 0, 1, 0, 0, 0])
    confirmed = pd.DataFrame(
        {"long_confirmed": candidates == 1, "short_confirmed": candidates == -1},
        index=index,
    )
    result = strategy.generate_signals(data, confirmed, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat = pd.concat([data] * 10, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=len(flat), freq="D")
    derived = strategy.indicators(flat, params)
    assert not derived["long_confirmed"].any()
    assert not derived["short_confirmed"].any()
    expected_long = derived["up_overflow"] & derived["up_overflow"].shift(1, fill_value=False)
    expected_short = derived["down_overflow"] & derived["down_overflow"].shift(1, fill_value=False)
    assert derived["long_confirmed"].equals(expected_long)
    assert derived["short_confirmed"].equals(expected_short)


if __name__ == "__main__":
    _self_check()
