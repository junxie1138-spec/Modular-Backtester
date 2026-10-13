from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    forecast_bars: int = 7
    entry_score: float = 0.20
    min_coarse_share: float = 0.40


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791328878"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if not 5 <= params.forecast_bars <= 10:
            raise ValueError("forecast_bars must be between 5 and 10")
        if not np.isfinite(params.entry_score) or params.entry_score <= 0:
            raise ValueError("entry_score must be finite and positive")
        if not 0 <= params.min_coarse_share <= 1:
            raise ValueError("min_coarse_share must be between 0 and 1")
        out = pd.DataFrame(index=data.index)
        r = np.log(data["close"].where(data["close"] > 0)).diff()
        r = r.replace([np.inf, -np.inf], np.nan)
        out["return"] = r
        for lag in (1, 2, 3, 4):
            out[f"lag_{lag}"] = r.shift(lag)
        # Causal Haar details on overlapping blocks of 2, 4 and 8 returns.
        out["haar_1"] = (r - r.shift(1)) / np.sqrt(2.0)
        for scale in (2, 4):
            block = r.rolling(scale, min_periods=scale).sum()
            out[f"block_{scale}"] = block
            out[f"haar_{scale}"] = (block - block.shift(scale)) / np.sqrt(2.0 * scale)
        for scale in (1, 2, 4):
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2)
            out[f"rho_{scale}"] = r.rolling(4, min_periods=4).corr(r.shift(scale)).clip(-1.0, 1.0).fillna(0.0)
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 1e-16)
        for scale in (1, 2, 4):
            out[f"weight_{scale}"] = (out[f"energy_{scale}"] / denominator).fillna(0.0)
        out["rms"] = np.sqrt(r.pow(2).rolling(8, min_periods=8).mean())
        out["ready"] = (out["total_energy"].notna() & out["rms"].gt(0)).astype(int)
        history = {0: r, -1: r.shift(1), -2: r.shift(2), -3: r.shift(3)}
        out["forecast_total"] = 0.0
        for step in range(1, params.forecast_bars + 1):
            prediction = sum(
                out[f"weight_{scale}"] * out[f"rho_{scale}"] * history[step - scale]
                for scale in (1, 2, 4)
            )
            history[step] = 0.85 * prediction
            out[f"forecast_{step}"] = history[step]
            out["forecast_total"] = out["forecast_total"] + history[step]
        out["score"] = out["forecast_total"] / (out["rms"].where(out["rms"] > 0) * np.sqrt(params.forecast_bars))
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        eligible = indicators["ready"].gt(0) & indicators["weight_4"].ge(params.min_coarse_share)
        candidates = np.select(
            [eligible & indicators["score"].ge(params.entry_score),
             eligible & indicators["score"].le(-params.entry_score)],
            [1, -1], default=0,
        )
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        for bar, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            decisions[bar] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = EchoParams()
    index = pd.date_range("2020-01-01", periods=16)
    close = pd.Series(100.0 * np.exp(np.arange(16) * 0.002 + 0.01 * np.sin(np.arange(16))), index=index)
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0})
    features = strategy.indicators(data, params)
    assert np.isfinite(features.to_numpy()).all()
    assert features["ready"].iloc[:8].eq(0).all()
    changed = data.copy()
    changed.loc[index[12]:, "close"] *= 1.5
    pd.testing.assert_frame_equal(features.iloc[:12], strategy.indicators(changed, params).iloc[:12])
    flat = data.copy()
    flat["close"] = 100.0
    flat_features = strategy.indicators(flat, params)
    assert np.isfinite(flat_features.to_numpy()).all()
    assert strategy.generate_signals(flat, flat_features, None, params).data["signal"].eq(0).all()
    probe = pd.DataFrame({"ready": 1, "weight_4": 1.0, "score": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0]}, index=index[:8])
    result = strategy.generate_signals(data.iloc[:8], probe, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
