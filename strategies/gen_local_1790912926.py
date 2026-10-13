from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    fast_window: int = 10
    slow_window: int = 60
    rank_window: int = 126
    shock_window: int = 20
    entry_percentile: float = 0.60
    target_volatility: float = 0.12
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790912926"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(
            200,
            max(params.fast_window, params.slow_window)
            + params.rank_window + params.shock_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.fast_window, params.slow_window, params.rank_window, params.shock_window) < 2 or params.max_hold < 1:
            raise ValueError("Lookbacks must be at least two and max_hold positive.")
        if not 0.0 < params.entry_percentile < 0.90:
            raise ValueError("entry_percentile must be between zero and 0.90.")
        if not np.isfinite(params.target_volatility) or params.target_volatility <= 0.0:
            raise ValueError("target_volatility must be finite and positive.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0.0))
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["fast_volatility"] = out["return"].rolling(params.fast_window).std(ddof=0) * np.sqrt(252.0)
        out["slow_volatility"] = out["return"].rolling(params.slow_window).std(ddof=0) * np.sqrt(252.0)
        out["volatility_ratio"] = out["fast_volatility"] / out["slow_volatility"].where(out["slow_volatility"] > 0.0)
        out["volatility_rank"] = out["volatility_ratio"].rolling(params.rank_window).rank(pct=True)
        out["previous_rank"] = out["volatility_rank"].shift(1)
        out["recent_shock_rank"] = out["previous_rank"].rolling(params.shock_window).max()
        out["ma200"] = close.rolling(200).mean()
        out["target_size"] = (
            params.target_volatility / out["fast_volatility"].clip(lower=0.01)
        ).clip(lower=0.05, upper=1.0).fillna(0.05)

        relaxed = (
            (out["recent_shock_rank"] >= 0.90)
            & (out["previous_rank"] > params.entry_percentile)
            & (out["volatility_rank"] <= params.entry_percentile)
        )
        out["entry_direction"] = np.select(
            [relaxed & (close > out["ma200"]), relaxed & (close < out["ma200"])],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw_signal = np.zeros(len(data), dtype=int)
        raw_size = np.ones(len(data), dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        position_size = 1.0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["entry_direction"].to_numpy(dtype=int)
        sizes = indicators["target_size"].to_numpy(dtype=float)

        for bar in range(len(data)):
            if direction:
                if bar == entry_bar:
                    entry_price = opens[bar]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = (
                    direction * (closes[bar] / entry_price - 1.0)
                    if valid_entry and np.isfinite(closes[bar]) else np.nan
                )
                if (
                    not valid_entry
                    or gain >= params.profit_target
                    or bar - entry_bar + 1 >= params.max_hold
                ):
                    direction = 0
                    entry_price = np.nan
            elif candidates[bar] != 0 and np.isfinite(closes[bar]) and closes[bar] > 0.0:
                direction = int(candidates[bar])
                entry_bar = bar + 1
                position_size = float(sizes[bar])

            raw_signal[bar] = direction
            raw_size[bar] = position_size if direction else 1.0

        df = pd.DataFrame({"signal": raw_signal, "size": raw_size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({
        "open": 100.0,
        "high": 106.0,
        "low": 94.0,
        "close": [100.0, 100.0, 105.0, 100.0, 95.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry_direction": [1, 0, 0, -1, 0, 0],
        "target_size": 0.5,
    }, index=index)
    strategy = GeneratedStrategy()
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, context, GeneratedParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert result["size"].iloc[1] == 0.5
    assert np.isfinite(result["size"]).all() and result["size"].gt(0.0).all()
    data["close"] = 100.0
    timed = strategy.generate_signals(data, indicators, context, GeneratedParams(max_hold=2)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    warmup = strategy.indicators(data, GeneratedParams())
    assert warmup["entry_direction"].eq(0).all()
