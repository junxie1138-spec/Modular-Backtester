from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    peak_window: int = 40
    range_window: int = 20
    box_window: int = 3
    min_drawdown: float = 0.02
    max_drawdown: float = 0.08
    compression_limit: float = 1.35
    deepening_tolerance: float = 0.15
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791208292"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return max(params.peak_window, params.range_window, params.box_window) + 2

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = data["high"].rolling(params.peak_window).max()
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0) / close
        out["normal_range"] = out["daily_range"].rolling(params.range_window).median()
        out["drawdown"] = 1.0 - close / out["peak"].where(out["peak"] > 0)
        out["box_span"] = (
            data["high"].rolling(params.box_window).max()
            - data["low"].rolling(params.box_window).min()
        ) / close
        out["compression"] = out["box_span"] / out["normal_range"].where(out["normal_range"] > 0)
        out["deepening"] = out["drawdown"].diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            out["drawdown"].between(params.min_drawdown, params.max_drawdown)
            & out["compression"].le(params.compression_limit)
            & out["deepening"].le(params.deepening_tolerance * out["normal_range"])
        ).fillna(False)
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: CompressionParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        pending_entry = False
        armed = True
        for i in range(len(data)):
            if pending_entry:
                # The prior close's decision fills at this bar's open.
                entry_bar = i
                entry_price = opens[i]
                pending_entry = False
            if entry_bar >= 0:
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_up = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_up:
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif confirmed[i] and armed:
                signals[i] = 1
                pending_entry = True
                armed = False
            if not confirmed[i]:
                armed = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    forced = pd.DataFrame({"confirmed": [False, True] + [False] * 8}, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[3], "close"] = 102.0
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    short = strategy.indicators(data, params)
    assert not short["confirmed"].any()
    assert strategy.warmup_bars(params) == 42
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
