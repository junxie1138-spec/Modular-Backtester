from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuietAnchorParams:
    anchor_window: int = 10
    fast_atr_window: int = 3
    slow_atr_window: int = 14
    stretch_threshold: float = 1.2
    compression_limit: float = 0.9


class GeneratedStrategy(BaseStrategy[QuietAnchorParams]):
    strategy_id = "gen_local-w2_1791255416"

    @classmethod
    def params_type(cls) -> type[QuietAnchorParams]:
        return QuietAnchorParams

    def warmup_bars(self, params: QuietAnchorParams) -> int:
        return int(max(params.anchor_window, params.fast_atr_window, params.slow_atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: QuietAnchorParams) -> pd.DataFrame:
        if min(params.anchor_window, params.fast_atr_window, params.slow_atr_window) < 1:
            raise ValueError("Lookback windows must be positive")
        if not np.isfinite(params.stretch_threshold) or params.stretch_threshold <= 0:
            raise ValueError("stretch_threshold must be finite and positive")
        if not np.isfinite(params.compression_limit) or not 0 < params.compression_limit < 1:
            raise ValueError("compression_limit must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        out["quiet_weight"] = 1.0 / out["true_range"].where(out["true_range"] > 0)
        out["weighted_price"] = out["typical_price"] * out["quiet_weight"]
        out["weight_sum"] = out["quiet_weight"].rolling(params.anchor_window, min_periods=params.anchor_window).sum()
        out["price_sum"] = out["weighted_price"].rolling(params.anchor_window, min_periods=params.anchor_window).sum()
        out["quiet_anchor"] = out["price_sum"] / out["weight_sum"].where(out["weight_sum"] > 0)
        out["fast_atr"] = out["true_range"].rolling(params.fast_atr_window, min_periods=params.fast_atr_window).mean()
        out["slow_atr"] = out["true_range"].rolling(params.slow_atr_window, min_periods=params.slow_atr_window).mean()
        out["compression"] = out["fast_atr"] / out["slow_atr"].where(out["slow_atr"] > 0)
        out["stretch"] = (data["close"] - out["quiet_anchor"]) / out["fast_atr"].where(out["fast_atr"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QuietAnchorParams) -> SignalFrame:
        valid = np.isfinite(indicators["stretch"]) & np.isfinite(indicators["compression"])
        compressed = valid & indicators["compression"].le(params.compression_limit)
        raw = pd.Series(0, index=data.index, dtype=int)
        raw.loc[compressed & indicators["stretch"].le(-params.stretch_threshold)] = 1
        raw.loc[compressed & indicators["stretch"].ge(params.stretch_threshold)] = -1
        confirmed = raw.where(raw.eq(raw.shift(1)), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (candidate, entry) in enumerate(zip(raw.to_numpy(), confirmed.to_numpy())):
            if position and candidate == -position:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = QuietAnchorParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"stretch": [-2.0, -2.0, np.nan, 0.0, 2.0, 2.0, 2.0, 0.0, -2.0, -2.0], "compression": 0.8}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["stretch"].isna().all()
    assert strategy.warmup_bars(params) == 16


if __name__ == "__main__":
    _self_check()
