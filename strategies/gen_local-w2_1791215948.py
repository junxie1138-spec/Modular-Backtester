from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    inside_bars: int = 2
    target_pct: float = 0.006


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791215948"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(params.inside_bars) + 2

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if params.inside_bars < 1 or int(params.inside_bars) != params.inside_bars:
            raise ValueError("inside_bars must be a positive integer")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(
            (data["high"] > data["low"]) & (data["low"] > 0)
        )
        out["inside"] = (
            out["range"].notna()
            & out["range"].shift(1).notna()
            & data["high"].lt(data["high"].shift(1))
            & data["low"].gt(data["low"].shift(1))
        )
        groups = (~out["inside"]).cumsum()
        out["streak"] = out["inside"].astype(int).groupby(groups).cumsum().clip(
            upper=params.inside_bars
        )
        window = int(params.inside_bars) + 1
        out["ceiling"] = data["high"].rolling(window, min_periods=window).max().shift(1)
        out["entry"] = (
            out["streak"].shift(1).ge(params.inside_bars)
            & data["close"].gt(out["ceiling"])
            & out["range"].gt(out["range"].shift(1))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        if not np.isfinite(params.target_pct) or params.target_pct <= 0:
            raise ValueError("target_pct must be positive and finite")
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if active:
                # The previous close requested entry; today's open is now known.
                if held_bars == 0:
                    entry_price = opens[i]
                held_bars += 1
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.target_pct)
                )
                # Close-observed exits execute at the following open.
                if invalid_entry or target_hit or held_bars >= 2:
                    active = False
                    held_bars = 0
                else:
                    signals[i] = 1
            elif entries[i]:
                active = True
                held_bars = 0
                signals[i] = 1
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    bars = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 105.5, 105, 105],
            "high": [105, 104, 103, 106, 107, 107, 107],
            "low": [95, 96, 97, 99, 100, 100, 100],
            "close": [100, 100, 100, 105.5, 106, 105, 105],
            "volume": [1000] * 7,
        },
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for first_close, expected in (
        (106.0, [0, 0, 0, 0, 1, 1, 0]),
        (106.5, [0, 0, 0, 0, 1, 0, 0]),
    ):
        bars.iloc[4, bars.columns.get_loc("close")] = first_close
        derived = strategy.indicators(bars, params)
        result = strategy.generate_signals(bars, derived, ctx, params).data
        assert result["signal"].tolist() == expected
        for n in range(1, len(bars) + 1):
            prefix = bars.iloc[:n]
            actual = strategy.generate_signals(
                prefix, strategy.indicators(prefix, params), ctx, params
            ).data
            assert actual["signal"].tolist() == expected[:n]
    missing = bars * np.nan
    result = strategy.generate_signals(
        missing, strategy.indicators(missing, params), ctx, params
    ).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()
