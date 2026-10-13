from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    rank_window: int = 40
    analog_window: int = 120
    match_percentile: float = 0.25
    entry_percentile: float = 0.80
    hold_bars: int = 4
    profit_target: float = 0.0125

    def __post_init__(self):
        if not (
            self.rank_window >= 5 and self.analog_window >= 20
            and 3 <= self.hold_bars <= 5
            and 0 < self.match_percentile <= 1
            and 0.5 < self.entry_percentile < 1
            and np.isfinite(self.profit_target) and self.profit_target > 0
        ):
            raise ValueError("Invalid analog parameters")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790850195"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(2 * params.rank_window + params.analog_window + params.hold_bars + 4)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].where(data["volume"] >= 0)
        features = pd.DataFrame(index=data.index)
        features["return"] = returns
        features["return_rank"] = returns.rolling(params.rank_window).rank(pct=True)
        features["volume_rank"] = volume.rolling(params.rank_window).rank(pct=True)
        shape = {}
        for k in range(3):
            shape[f"return_rank_{k}"] = features["return_rank"].shift(k)
            shape[f"volume_rank_{k}"] = features["volume_rank"].shift(k)
        shape = pd.DataFrame(shape, index=data.index)
        distances, outcomes = {}, {}
        # Labels finish before the current three-bar shape begins.
        first_lag = params.hold_bars + 3
        for lag in range(first_lag, first_lag + params.analog_window):
            delta = (shape - shape.shift(lag)).abs()
            distances[lag] = delta.sum(axis=1, min_count=6) / 6.0
            outcomes[lag] = close.shift(lag - params.hold_bars) / opening.shift(lag - 1) - 1.0
        distances = pd.DataFrame(distances, index=data.index)
        outcomes = pd.DataFrame(outcomes, index=data.index).replace([np.inf, -np.inf], np.nan)
        valid = distances.notna() & outcomes.notna()
        cutoff = distances.where(valid).quantile(params.match_percentile, axis=1)
        matches = valid & distances.le(cutoff, axis=0)
        score = outcomes.where(matches).mean(axis=1)
        score = score.where(valid.sum(axis=1).eq(params.analog_window))
        features["distance_cutoff"] = cutoff
        features["analog_count"] = matches.sum(axis=1)
        features["forecast"] = score
        features["forecast_rank"] = score.rolling(params.rank_window).rank(pct=True)
        return pd.concat(
            [features, shape, distances.add_prefix("distance_"),
             outcomes.add_prefix("outcome_"), matches.add_prefix("match_")], axis=1
        )

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"]
        rank = indicators["forecast_rank"]
        entries = np.select(
            [(rank >= params.entry_percentile) & (forecast > 0),
             (rank <= 1.0 - params.entry_percentile) & (forecast < 0)],
            [1, -1], default=0,
        )
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side, entry_bar, entry_price = 0, -1, np.nan
        warmup = max(self.warmup_bars(params), int(ctx.warmup_bars))
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0
                else:
                    gain = side * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        side = 0
            elif i >= warmup - 1 and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                if side:
                    entry_bar, entry_price = i + 1, np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AnalogParams()
    start = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=start + 140, freq="B")
    rng = np.random.default_rng(19)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, len(index))))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.integers(1000, 10000, len(index))},
        index=index,
    )
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-20], params)
    pd.testing.assert_frame_equal(full.iloc[:-20], prefix)
    ctx = StrategyContext("SPY", "1d", start)
    actual = strategy.generate_signals(data, full, ctx, params).data
    assert set(actual["signal"].unique()) <= {-1, 0, 1}
    assert actual["size"].gt(0).all()
    assert actual["signal"].iloc[:start].eq(0).all()
    data.loc[:, ["open", "close"]] = 100.0
    mock = pd.DataFrame({"forecast": 0.0, "forecast_rank": 0.5}, index=index)
    mock.iloc[start] = [0.01, 1.0]
    mock.iloc[start + 7] = [-0.01, 0.0]
    data.iloc[start + 2, data.columns.get_loc("close")] = 102.0
    signal = strategy.generate_signals(data, mock, ctx, params).data["signal"]
    assert signal.iloc[start:start + 4].tolist() == [0, 1, 1, 0]
    assert signal.iloc[start + 7:start + 13].tolist() == [0, -1, -1, -1, -1, 0]
    data.iloc[start + 9, data.columns.get_loc("close")] = 98.0
    signal = strategy.generate_signals(data, mock, ctx, params).data["signal"]
    assert signal.iloc[start + 8:start + 11].tolist() == [-1, -1, 0]


if __name__ == "__main__":
    _self_check()
