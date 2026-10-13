from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compression_window: int = 3
    baseline_window: int = 20
    compression_ratio: float = 0.9


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791336061"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.compression_window, params.baseline_window) + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.compression_window < 2 or params.baseline_window < 2 or not np.isfinite(params.compression_ratio) or params.compression_ratio <= 0:
            raise ValueError("Windows must be at least two and compression_ratio must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["prior_return"] = out["return"].shift(1)
        out["baseline"] = out["return"].abs().shift(1).rolling(params.baseline_window).mean().replace(0.0, np.nan)
        out["return_width"] = out["return"].rolling(params.compression_window).max() - out["return"].rolling(params.compression_window).min()
        out["compression"] = out["return_width"] / out["baseline"]
        out["recoil"] = out["return"] - out["prior_return"]
        out["loading"] = -(out["return"] + out["prior_return"]) / out["baseline"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entry = (
            (indicators["compression"] <= params.compression_ratio)
            & (indicators["return"] < 0)
            & (indicators["prior_return"] < 0)
            & (indicators["recoil"] > 0)
        ).fillna(False)
        strength = (
            indicators["loading"].clip(0.0, 2.0) / 2.0
        ) * (1.0 - indicators["compression"] / params.compression_ratio).clip(0.0, 1.0)
        df = pd.DataFrame(index=data.index)
        # Remain long exactly while entry holds; its reversal is the sole exit.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        # Exposure uses the same decision bar as the shifted signal.
        df["size"] = (0.2 + 0.75 * strength.fillna(0.0)).shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    returns = np.r_[np.tile([-0.01, 0.01], 12), [-0.004, -0.002, -0.001, 0.003, 0.001, -0.003, -0.002, -0.001, 0.002, 0.001]]
    close = 100.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
    data = pd.DataFrame(
        {column: close for column in ("open", "high", "low", "close")},
        index=pd.date_range("2020-01-01", periods=len(close)),
    )
    data["volume"] = 1000.0
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert result["signal"].iloc[[27, 28, 29, 32, 33, 34]].tolist() == [0, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all() and result["size"].le(0.95).all()
    assert result.loc[result["signal"].eq(1), "size"].nunique() > 1
    prefix = data.iloc[:33]
    pd.testing.assert_frame_equal(
        result.iloc[:33],
        strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data,
    )
