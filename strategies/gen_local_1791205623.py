from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    support_window: int = 10
    peak_window: int = 40
    percentile_window: int = 80
    drawdown_quantile: float = 0.65
    volume_quantile: float = 0.75
    atr_window: int = 14
    atr_multiplier: float = 2.0
    hold_bars: int = 4
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791205623"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(
            params.peak_window + params.percentile_window + 1,
            params.support_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["prior_floor"] = data["low"].shift(1).rolling(params.support_window).min()
        out["peak"] = data["close"].rolling(params.peak_window).max()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].replace(0.0, np.nan)
        out["previous_drawdown"] = out["drawdown"].shift(1)
        out["drawdown_cutoff"] = out["previous_drawdown"].rolling(
            params.percentile_window
        ).quantile(params.drawdown_quantile)
        out["volume_cutoff"] = data["volume"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.volume_quantile)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry"] = (
            (out["previous_drawdown"] > 0.0)
            & (out["previous_drawdown"] >= out["drawdown_cutoff"])
            & (data["low"] < out["prior_floor"])
            & (data["close"] > out["prior_floor"])
            & (data["close"] > out["previous_close"])
            & (data["volume"] >= out["volume_cutoff"])
            & (data["volume"] > 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0.0
            if active:
                held += 1
                if valid:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.atr_multiplier * atrs[i])
                if not valid or closes[i] <= stop or held >= params.hold_bars:
                    active = False
                raw[i] = int(active)
            elif valid and entries[i]:
                active = True
                held = 0
                # The first in-trade close is next bar, after the shifted entry.
                high_water = stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100.0, 103.0, 102.0, 100.5, 100.5, 100.5]}, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": [1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    indicators["atr"] = 1.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert signals["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
