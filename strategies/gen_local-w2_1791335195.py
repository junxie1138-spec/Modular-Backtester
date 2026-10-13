from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalGapParams:
    history: int = 504
    atr_period: int = 14
    hold_bars: int = 18
    min_observations: int = 12
    gap_floor: float = 0.25
    entry_score: float = 0.6
    stop_atr: float = 2.8
    full_size_score: float = 2.0
    min_size: float = 0.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SeasonalGapParams]):
    strategy_id = "gen_local-w2_1791335195"

    @classmethod
    def params_type(cls):
        return SeasonalGapParams

    def warmup_bars(self, params: SeasonalGapParams) -> int:
        return int(params.history + params.hold_bars + params.atr_period + 1)

    def indicators(self, data: pd.DataFrame, params: SeasonalGapParams) -> pd.DataFrame:
        if not (
            params.history >= params.min_observations >= 3
            and params.atr_period >= 2
            and 15 <= params.hold_bars <= 20
            and params.gap_floor > 0
            and params.entry_score > 0
            and params.stop_atr > 0
            and params.full_size_score > 0
            and 0 < params.min_size <= params.max_size <= 1
        ):
            raise ValueError("Invalid seasonal gap parameters")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["gap_atr"] = (
            (data["open"] - previous_close) / out["atr"].shift(1).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        phase = np.minimum((data.index.day - 1) // 10, 2)
        out["bucket"] = phase * 5 + data.index.dayofweek
        out["matured_bucket"] = out["bucket"].shift(params.hold_bars)
        out["matured_gap"] = out["gap_atr"].shift(params.hold_bars)
        out["matured_move"] = (
            (data["close"] - data["close"].shift(params.hold_bars))
            / out["atr"].shift(params.hold_bars).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["beta"] = np.nan
        out["observations"] = 0.0
        valid = out["matured_gap"].notna() & out["matured_move"].notna()
        for bucket in range(15):
            sample = valid & out["matured_bucket"].eq(bucket)
            x = out["matured_gap"].where(sample, 0.0)
            y = out["matured_move"].where(sample, 0.0)
            count = sample.astype(float).rolling(params.history).sum()
            denominator = x.pow(2).rolling(params.history).sum().replace(0, np.nan)
            beta = (x * y).rolling(params.history).sum() / denominator
            current = out["bucket"].eq(bucket)
            out.loc[current, "beta"] = beta.loc[current]
            out.loc[current, "observations"] = count.loc[current]
        out["score"] = (out["beta"] * out["gap_atr"]).where(
            out["observations"].ge(params.min_observations), 0.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalGapParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position, entry_bar = 0, 0
        extreme, stop, entry_size = 0.0, 0.0, params.min_size
        for i, (close, atr, gap, score) in enumerate(zip(
            data["close"].to_numpy(), indicators["atr"].to_numpy(),
            indicators["gap_atr"].to_numpy(), indicators["score"].to_numpy(),
        )):
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                continue
            if position:
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.stop_atr * atr)
                    stopped = close <= stop
                else:
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.stop_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.hold_bars:
                    position = 0
                else:
                    signals[i], sizes[i] = position, entry_size
                continue
            if (
                np.isfinite(gap) and np.isfinite(score)
                and abs(gap) >= params.gap_floor
                and abs(score) >= params.entry_score
            ):
                position = 1 if score > 0 else -1
                entry_bar, extreme = i, close
                stop = close - position * params.stop_atr * atr
                entry_size = float(np.clip(
                    abs(score) / params.full_size_score,
                    params.min_size, params.max_size,
                ))
                signals[i], sizes[i] = position, entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SeasonalGapParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    data = pd.DataFrame({"close": [100, 103, 104, 102, 100, 99, 98, 100, 101, 102]}, index=index)
    derived = pd.DataFrame({
        "atr": [1, 1, 1, 5, 1, 1, 1, 5, 1, 1],
        "gap_atr": [1.0] * 10,
        "score": [1, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, 0, 0]
    assert result["size"].iloc[1] == 0.5
    assert result["size"].gt(0).all()
