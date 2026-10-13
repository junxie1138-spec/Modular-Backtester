from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SlopeShockParams:
    trend_bars: int = 20
    noise_bars: int = 30
    entry_z: float = 1.5
    profit_target: float = 0.012
    hold_bars: int = 4
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SlopeShockParams]):
    strategy_id = "gen_local_1791318168"

    @classmethod
    def params_type(cls):
        return SlopeShockParams

    def warmup_bars(self, params: SlopeShockParams) -> int:
        return params.trend_bars + params.noise_bars + 1

    def indicators(self, data: pd.DataFrame, params: SlopeShockParams) -> pd.DataFrame:
        if (params.trend_bars < 3 or params.noise_bars < 2
                or params.entry_z <= 0 or params.profit_target <= 0
                or not 3 <= params.hold_bars <= 5
                or not 0 < params.max_size <= 1):
            raise ValueError("Invalid slope-shock parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["trend_mean"] = out["log_close"].rolling(params.trend_bars).mean()
        weights = np.arange(params.trend_bars, dtype=float)
        weights -= weights.mean()
        weights /= np.dot(weights, weights)
        out["slope"] = np.nan
        if len(data) >= params.trend_bars:
            out.iloc[params.trend_bars - 1:, out.columns.get_loc("slope")] = np.convolve(
                out["log_close"].to_numpy(), weights[::-1], mode="valid"
            )
        out["slope_innovation"] = out["slope"].diff()
        out["innovation_scale"] = (
            out["slope_innovation"].rolling(params.noise_bars).std(ddof=0).shift(1)
        )
        out["innovation_z"] = (
            out["slope_innovation"] / out["innovation_scale"].where(
                out["innovation_scale"] > 1e-12
            )
        ).replace([np.inf, -np.inf], np.nan)
        out["prior_slope"] = out["slope"].shift(1)
        out["trend_endpoint"] = (
            out["trend_mean"] + out["slope"] * (params.trend_bars - 1) / 2
        )
        out["trend_residual"] = out["log_close"] - out["trend_endpoint"]
        eligible = (out["prior_slope"] > 0) & (out["trend_residual"] < 0)
        out["entry_strength"] = (
            (-out["innovation_z"] / params.entry_z - 1).clip(0, 1)
            .where(eligible, 0).fillna(0)
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SlopeShockParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": float(params.max_size)}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.max_size, dtype=float)
        strength = indicators["entry_strength"].fillna(0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        position_size = params.max_size
        for i in range(len(data)):
            if entry_bar >= 0:
                # The previous close's entry fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        entry_bar = -1
                        continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= params.hold_bars
                if target_hit or timed_out:
                    # Exit decision at this close; execution is next session.
                    entry_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif np.isfinite(strength[i]) and strength[i] > 0:
                entry_bar = i + 1
                entry_price = np.nan
                position_size = params.max_size * (0.5 + 0.5 * min(strength[i], 1.0))
                signals[i] = 1
                sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SlopeShockParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.iloc[7, data.columns.get_loc("close")] = 102.0
    indicators = pd.DataFrame(
        {"entry_strength": [0.2, 0, 0, 0, 0, 0.8, 0, 0, 0, 0]}, index=index
    )
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.57)
    assert np.isclose(result["size"].iloc[6], 0.855)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
