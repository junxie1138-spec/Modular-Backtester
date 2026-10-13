from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entropy_window: int = 18
    history_bars: int = 504
    complexity_limit: float = 0.92
    entry_score: float = 0.8
    min_samples: int = 8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791238886"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.history_bars + params.entropy_window + params.hold_bars + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (
            params.entropy_window < 3 or params.history_bars < 20
            or params.min_samples < 2 or not 3 <= params.hold_bars <= 5
            or not 0.0 < params.complexity_limit <= 1.0
            or not np.isfinite(params.entry_score) or params.entry_score <= 0.0
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        r = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = r
        a, b, c = r.shift(2), r.shift(1), r
        valid = a.notna() & b.notna() & c.notna()
        code = ((a > b).astype(int) * 4 + (a > c).astype(int) * 2
                + (b > c).astype(int)).where(valid)
        out["ordinal_code"] = code
        probabilities = pd.DataFrame(index=data.index)
        for pattern in (0, 1, 3, 4, 6, 7):
            probability = code.eq(pattern).astype(float).where(valid).rolling(
                params.entropy_window, min_periods=params.entropy_window
            ).mean()
            probabilities[pattern] = probability
            out[f"pattern_probability_{pattern}"] = probability
        out["complexity"] = -(probabilities * np.log(
            probabilities.clip(lower=1e-12)
        )).sum(axis=1, min_count=6) / np.log(6.0)
        out["eligible"] = out["complexity"].le(params.complexity_limit)
        out["realized_return"] = data["close"].div(
            data["close"].shift(params.hold_bars)
        ).sub(1.0).replace([np.inf, -np.inf], np.nan)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["weekday"] = weekday
        out["decision_weekday"] = weekday.shift(params.hold_bars)
        matured = out["eligible"].shift(params.hold_bars, fill_value=False)
        outcome = out["realized_return"]
        score = pd.Series(np.nan, index=data.index, dtype=float)
        for day in range(5):
            sample = matured & out["decision_weekday"].eq(day) & outcome.notna()
            count = sample.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            total = outcome.where(sample, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            squares = outcome.pow(2).where(sample, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            mean = total / count.replace(0.0, np.nan)
            variance = ((squares - total.pow(2) / count.replace(0.0, np.nan))
                        / (count - 1.0).where(count > 1.0)).clip(lower=0.0)
            stderr = np.sqrt(variance / count.replace(0.0, np.nan))
            estimate = (mean / stderr.replace(0.0, np.nan)).where(
                count >= params.min_samples
            ).replace([np.inf, -np.inf], np.nan)
            out[f"sample_count_{day}"] = count
            out[f"seasonal_mean_{day}"] = mean
            out[f"seasonal_stderr_{day}"] = stderr
            out[f"seasonal_score_{day}"] = estimate
            score = score.where(weekday.ne(day), estimate)
        ready = np.arange(len(data)) >= self.warmup_bars(params) - 1
        out["score"] = score.where(out["eligible"] & ready)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        scores = indicators["score"].fillna(0.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        position, exit_bar, entry_size = 0, -1, 0.2
        for i, score in enumerate(scores):
            if position:
                if i == exit_bar:
                    position = 0
                    continue
            elif np.isfinite(score) and abs(score) >= params.entry_score:
                position = 1 if score > 0.0 else -1
                exit_bar = i + params.hold_bars
                entry_size = float(np.clip(
                    0.2 + 0.75 * abs(score) / (abs(score) + params.entry_score),
                    0.2, 0.95
                ))
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(index=index)
    scores = pd.DataFrame({"score": [1.0] + [-2.0] * 11}, index=index)
    result = GeneratedStrategy().generate_signals(data, scores, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    assert result["size"].gt(0.0).all()
    assert result["size"].iloc[1:5].nunique() == 1
    assert result["size"].iloc[6] > result["size"].iloc[1]
