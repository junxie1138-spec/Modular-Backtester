from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    slope_days: int = 5
    season_days: int = 126
    entry_z: float = 0.03


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791280701"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.season_days + params.slope_days + 3, 21))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.slope_days < 1 or params.season_days < 40 or not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("Require slope_days >= 1, season_days >= 40 and finite entry_z > 0.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        out["latent_slope"] = out["return"].rolling(params.slope_days).mean().shift(1)
        out["innovation"] = out["return"] - out["latent_slope"]
        out["volatility"] = out["return"].rolling(20).std(ddof=0).replace(0.0, np.nan)
        out["ma200"] = close.rolling(200).mean()
        out["weekday"] = data.index.dayofweek
        # On bar t, training pairs end at t-1: innovation[t-2], return[t-1].
        out["train_x"] = out["innovation"].shift(2)
        out["train_y"] = out["return"].shift(1)
        out["train_weekday"] = out["weekday"].shift(2)
        paired = out["train_x"].notna() & out["train_y"].notna()
        out["response"] = np.nan
        for day in range(5):
            mask = paired & out["train_weekday"].eq(day)
            out[f"count_{day}"] = mask.astype(float).rolling(params.season_days).sum()
            out[f"xx_{day}"] = out["train_x"].pow(2).where(mask, 0.0).rolling(params.season_days).sum()
            out[f"xy_{day}"] = (out["train_x"] * out["train_y"]).where(mask, 0.0).rolling(params.season_days).sum()
            denominator = out[f"xx_{day}"].where(out[f"xx_{day}"] > 1e-12)
            out[f"beta_{day}"] = (out[f"xy_{day}"] / denominator).where(out[f"count_{day}"] >= 8)
            out.loc[out["weekday"].eq(day), "response"] = out.loc[out["weekday"].eq(day), f"beta_{day}"]
        out["forecast"] = out["response"] * out["innovation"]
        out["score"] = (out["forecast"] / out["volatility"]).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        score = indicators["score"]
        direction = np.select(
            [score > params.entry_z, score < -params.entry_z],
            [1, -1],
            default=0,
        )
        regime = np.select(
            [data["close"] > indicators["ma200"], data["close"] < indicators["ma200"]],
            [1, -1],
            default=0,
        )
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(direction):
            if position:
                # Opposite seasonal evidence exits to flat; never reverse on the same bar.
                if candidate == -position:
                    position = 0
            elif candidate and candidate == regime[i]:
                position = int(candidate)
            signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    index = pd.bdate_range("2020-01-01", periods=340)
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index))))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:280], params)
    pd.testing.assert_frame_equal(full.iloc[:280], prefix)
    result = strategy.generate_signals(data, full, ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:280], strategy.generate_signals(data.iloc[:280], prefix, ctx, params).data)
    assert set(result["signal"].unique()) <= {-1, 0, 1}
    assert (result["size"] > 0).all()
    assert (result["signal"].iloc[:199] == 0).all()
    toy = data.iloc[:7].copy()
    toy["close"] = 100.0
    evidence = pd.DataFrame({"score": [1.0, np.nan, -1.0, -1.0, 0.0, 1.0, 1.0], "ma200": [99.0, 99.0, 101.0, 101.0, 101.0, 99.0, 99.0]}, index=toy.index)
    assert strategy.generate_signals(toy, evidence, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
