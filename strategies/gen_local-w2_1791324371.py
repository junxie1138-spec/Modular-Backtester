from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compression_ratio: float = 0.55
    profit_target: float = 0.006


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791324371"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not 0.0 < params.compression_ratio < 1.0:
            raise ValueError("compression_ratio must be between zero and one")
        if not 0.0 < params.profit_target < 1.0:
            raise ValueError("profit_target must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        absolute = out["return"].abs()
        out["quiet_amplitude"] = absolute.rolling(4, min_periods=4).mean().shift(1)
        out["baseline_amplitude"] = absolute.rolling(20, min_periods=20).mean().shift(1)
        out["quiet_peak"] = absolute.rolling(4, min_periods=4).max().shift(1)
        out["quiet_drift"] = out["return"].rolling(4, min_periods=4).mean().shift(1)
        out["compression"] = out["quiet_amplitude"].div(
            out["baseline_amplitude"].where(out["baseline_amplitude"] > 0)
        )
        out["cancellation"] = out["quiet_drift"].abs().div(
            out["quiet_amplitude"].where(out["quiet_amplitude"] > 0)
        )
        ready = (
            (out["compression"] <= params.compression_ratio)
            & (out["cancellation"] <= 0.25)
            & (absolute > out["quiet_peak"])
        )
        out["entry"] = np.select(
            [ready & (out["return"] > 0), ready & (out["return"] < 0)],
            [1, -1], default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # Observe the execution bar's open only after that bar arrives.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= 2:
                        position = 0
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=22)
    for direction in (1, -1):
        for target_hit in (False, True):
            data = pd.DataFrame(
                {"open": 100.0, "high": 102.0, "low": 98.0,
                 "close": 100.0, "volume": 1000.0}, index=index,
            )
            if target_hit:
                data.loc[index[1], "close"] = 100.0 * (1.0 + direction * 0.01)
            indicators = pd.DataFrame({"entry": [direction, 0, 0, 0, 0]}, index=index)
            result = strategy.generate_signals(data, indicators, ctx, params).data
            expected = [0, direction, 0, 0, 0] if target_hit else [0, direction, direction, 0, 0]
            assert result["signal"].tolist() == expected
            assert (result["size"] > 0).all()
    flat = strategy.indicators(data.assign(close=100.0), params)
    assert flat["entry"].eq(0).all()
