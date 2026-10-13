from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    bandwidth: float = 0.8
    innovation_weight: float = 2.0
    entry_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791247447"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (
            np.isfinite(params.bandwidth) and params.bandwidth > 0
            and np.isfinite(params.innovation_weight) and params.innovation_weight > 0
            and np.isfinite(params.entry_threshold) and params.entry_threshold > 0
        ):
            raise ValueError("All parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where((close > 0) & np.isfinite(close)))
        out["return"] = out["log_close"].diff()
        out["scale"] = np.sqrt(out["return"].pow(2).rolling(3, min_periods=3).mean()).clip(lower=1e-8)
        out["slope"] = (out["log_close"] - out["log_close"].shift(2)) / 2.0
        out["trend"] = out["slope"] / out["scale"]
        out["innovation"] = out["slope"].diff() / out["scale"]
        weights = []
        payoffs = []
        # ponytail: four recent analogs; a longer archive requires relaxing the ten-bar warmup.
        for lag in range(3, 7):
            out[f"trend_{lag}"] = out["trend"].shift(lag)
            out[f"innovation_{lag}"] = out["innovation"].shift(lag)
            out[f"outcome_{lag}"] = out["return"].shift(lag - 1) / out["scale"].shift(lag)
            out[f"distance_{lag}"] = (
                (out["trend"] - out[f"trend_{lag}"]).pow(2)
                + params.innovation_weight * (out["innovation"] - out[f"innovation_{lag}"]).pow(2)
            )
            out[f"weight_{lag}"] = 1.0 / (1.0 + out[f"distance_{lag}"] / params.bandwidth ** 2)
            out[f"payoff_{lag}"] = out[f"weight_{lag}"] * out[f"outcome_{lag}"]
            weights.append(f"weight_{lag}")
            payoffs.append(f"payoff_{lag}")
        out["total_weight"] = out[weights].sum(axis=1, min_count=4)
        out["forecast"] = out[payoffs].sum(axis=1, min_count=4) / out["total_weight"].where(out["total_weight"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"]
        entry = pd.Series(
            np.select(
                [forecast > params.entry_threshold, forecast < -params.entry_threshold],
                [1, -1], default=0,
            ), index=data.index, dtype=int,
        )
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, direction in enumerate(entry.to_numpy()):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams()
    index = pd.date_range("2020-01-01", periods=32, freq="D")
    close = 100.0 * np.exp(np.cumsum(0.003 * np.sin(np.arange(32) * 1.7)))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000000.0}, index=index,
    )
    indicators = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 10
    assert indicators["forecast"].iloc[:9].isna().all()
    assert np.isfinite(indicators["forecast"].iloc[9:]).all()
    pd.testing.assert_frame_equal(
        indicators.iloc[:20], strategy.indicators(data.iloc[:20], params),
    )
    probe = pd.DataFrame(
        {"forecast": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0]}, index=index[:7],
    )
    result = strategy.generate_signals(data.iloc[:7], probe, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    flat = data.copy()
    flat["close"] = 100.0
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
