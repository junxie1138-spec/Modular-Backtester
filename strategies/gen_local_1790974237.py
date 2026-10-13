from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    history_bars: int = 378
    percentile_window: int = 126
    entry_quantile: float = 0.8
    hold_bars: int = 8
    min_analogs: int = 3
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790974237"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.history_bars + params.percentile_window + params.hold_bars + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if (params.history_bars < 2 or params.percentile_window < 2
                or params.hold_bars < 1 or params.min_analogs < 1
                or not 0.0 < params.entry_quantile < 1.0
                or not np.isfinite(params.size) or params.size <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        direction = np.sign(data["close"].diff()).fillna(0).astype(int)
        groups = direction.ne(direction.shift(1)).cumsum()
        streak = direction.groupby(groups).cumcount().add(1).where(direction.ne(0), 0)
        boundary = direction.ne(direction.shift(1)) & direction.ne(0)
        reversal = boundary & direction.shift(1).fillna(0).ne(0)
        prior_run = streak.shift(1).where(boundary).ffill()
        older_run = prior_run.shift(1).where(boundary).ffill()
        out["direction"] = direction
        out["streak"] = streak
        out["reversal"] = reversal
        out["prior_run"] = prior_run
        out["older_run"] = older_run
        out["streak_threshold"] = streak.shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_quantile)
        out["matured_return"] = (
            data["open"].div(data["open"].shift(params.hold_bars)).sub(1)
        ).replace([np.inf, -np.inf], np.nan)
        valid_event = reversal & prior_run.gt(0) & older_run.gt(0)
        events = np.flatnonzero(valid_event.to_numpy())
        last = prior_run.to_numpy()
        older = older_run.to_numpy()
        directions = direction.to_numpy()
        outcomes = out["matured_return"].to_numpy()
        forecasts = np.full(len(data), np.nan)
        counts = np.zeros(len(data), dtype=int)
        # ponytail: bounded event scan; index run-length pairs if history grows materially.
        for i in events:
            if i < params.history_bars + params.hold_bars + 1:
                continue
            candidates = events[
                (events >= i - params.history_bars + 1)
                & (events + params.hold_bars + 1 <= i)
            ]
            matches = candidates[
                (last[candidates] == last[i])
                & (np.abs(older[candidates] - older[i]) <= 1)
            ]
            # j+1 entry open and j+hold_bars+1 exit open are both known by i.
            results = outcomes[matches + params.hold_bars + 1] * directions[matches]
            results = results[np.isfinite(results)]
            counts[i] = len(results)
            if len(results) >= params.min_analogs:
                forecasts[i] = directions[i] * results.mean()
        out["analog_count"] = counts
        out["forecast"] = forecasts
        out["confidence"] = out["forecast"].abs().ffill()
        out["confidence_threshold"] = out["confidence"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_quantile)
        eligible = (
            valid_event
            & out["prior_run"].ge(out["streak_threshold"])
            & out["forecast"].abs().ge(out["confidence_threshold"])
            & out["forecast"].notna()
            & out["forecast"].ne(0)
        )
        eligible.iloc[:self.warmup_bars(params)] = False
        out["entry"] = np.sign(out["forecast"].fillna(0)).where(eligible, 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position:
                if i == exit_bar:
                    position = 0
                signal[i] = position
                continue
            if entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=2 * params.hold_bars + 4)
    data = pd.DataFrame(index=index)
    entries = np.full(len(index), -1, dtype=int)
    entries[0] = 1
    indicators = pd.DataFrame({"entry": entries}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, params).data
    h = params.hold_bars
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:h + 1].eq(1).all()
    assert result["signal"].iloc[h + 1] == 0
    assert result["signal"].iloc[h + 2:2 * h + 2].eq(-1).all()
    assert result["signal"].iloc[2 * h + 2] == 0
    assert result["size"].gt(0).all()
