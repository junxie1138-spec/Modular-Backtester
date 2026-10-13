from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_window: int = 12
    percentile_window: int = 126
    tail_quantile: float = 0.8
    profit_target: float = 0.012
    max_hold: int = 4
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790975822"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if params.graph_window < 3 or params.percentile_window < 3:
            raise ValueError("Lookbacks must be at least three bars.")
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError("tail_quantile must be between 0.5 and 1.")
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between three and five bars.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")
        if not np.isfinite(params.size) or params.size <= 0:
            raise ValueError("size must be finite and positive.")
        return int(params.graph_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        x = out["log_close"]
        high_degree = pd.Series(0.0, index=data.index)
        low_degree = pd.Series(0.0, index=data.index)
        between_max = pd.Series(-np.inf, index=data.index)
        between_min = pd.Series(np.inf, index=data.index)
        # Horizontal visibility: endpoints must dominate every intervening node.
        # Applying the same graph to inverted prices identifies trough hubs.
        for lag in range(1, params.graph_window):
            past = x.shift(lag)
            high_degree += (np.minimum(x, past) > between_max).astype(float)
            low_degree += (np.maximum(x, past) < between_min).astype(float)
            between_max = np.maximum(between_max, past)
            between_min = np.minimum(between_min, past)
        valid = x.rolling(params.graph_window).count().eq(params.graph_window)
        out["high_degree"] = high_degree.where(valid)
        out["low_degree"] = low_degree.where(valid)
        out["drawdown"] = 1.0 - close / close.rolling(params.graph_window).max()
        out["runup"] = close / close.rolling(params.graph_window).min() - 1.0
        out["low_release"] = (-out["low_degree"].diff()).clip(lower=0)
        out["high_release"] = (-out["high_degree"].diff()).clip(lower=0)
        for name in ("high_degree", "low_degree", "drawdown", "runup",
                     "low_release", "high_release"):
            out[name + "_threshold"] = (
                out[name].shift(1).rolling(params.percentile_window)
                .quantile(params.tail_quantile)
            )
        long_entry = (
            (out["low_degree"].shift(1) > out["low_degree_threshold"].shift(1))
            & (out["drawdown"].shift(1) > out["drawdown_threshold"].shift(1))
            & (out["low_release"] > out["low_release_threshold"])
            & (close > close.shift(1))
        )
        short_entry = (
            (out["high_degree"].shift(1) > out["high_degree_threshold"].shift(1))
            & (out["runup"].shift(1) > out["runup_threshold"].shift(1))
            & (out["high_release"] > out["high_release_threshold"])
            & (close < close.shift(1))
        )
        out["entry"] = long_entry.astype(int) - short_entry.astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # The prior close's decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                # Targets are observed at closes; flat orders fill next open.
                if (not valid_price or gain >= params.profit_target
                        or i - entry_bar + 1 >= params.max_hold):
                    side = 0
            elif entries[i] in (-1, 1):
                side = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(graph_window=5, percentile_window=3, profit_target=0.02)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 110.0, "low": 90.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    forced = pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0, 0, 0]}, index=index)
    assert strategy.generate_signals(data, forced, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    for direction in (1, -1):
        forced.iloc[0, 0] = direction
        moved = data.copy()
        moved.iloc[2, moved.columns.get_loc("close")] = 100.0 + direction * 3.0
        result = strategy.generate_signals(moved, forced, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0, 0, 0]
        assert (result["size"] > 0).all()
    prices = [5.0, 2.0, 4.0, 1.0, 3.0, 2.5, 3.5, 4.0]
    graph_data = data.copy()
    graph_data["close"] = prices
    derived = strategy.indicators(graph_data, params)
    assert derived["high_degree"].iloc[4] == 2.0
    assert derived["low_degree"].iloc[4] == 1.0
    pd.testing.assert_frame_equal(
        derived.iloc[:6], strategy.indicators(graph_data.iloc[:6], params)
    )
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
