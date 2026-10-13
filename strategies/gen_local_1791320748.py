from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compression_quantile: float = 0.20
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791320748"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        # Three-bar aggregation followed by a 63-bar rank needs 65 bars.
        return 65

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not 0.0 < params.compression_quantile < 1.0:
            raise ValueError("compression_quantile must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14).mean()
        out["box_high"] = data["high"].rolling(3).max()
        out["box_low"] = data["low"].rolling(3).min()
        out["box_width"] = (out["box_high"] - out["box_low"]) / data["close"].where(data["close"] > 0.0)
        out["volume_sum"] = data["volume"].rolling(3).sum()
        out["width_rank"] = out["box_width"].rolling(63).rank(pct=True)
        out["volume_rank"] = out["volume_sum"].rolling(63).rank(pct=True)
        out = out.replace([np.inf, -np.inf], np.nan)
        # Heavy turnover in a small price box suggests absorption of supply.
        out["entry"] = (
            (out["width_rank"] <= params.compression_quantile)
            & (out["volume_rank"] >= 0.80)
            & (data["close"] > data["open"])
            & (data["close"] > out["previous_close"])
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
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0
            if active:
                age += 1
                if valid:
                    # First update is on the entry-fill bar, never the decision bar.
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or age >= 4:
                    active = False
                signal[i] = int(active)
            elif valid and entry[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [1000.0, 102.0, 101.0, 99.0, 105.0, 106.0]}, index=index)
    ind = pd.DataFrame({"atr": [1.0, 1.0, 10.0, 10.0, 1.0, 1.0],
                        "entry": [True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=65)
    # ATR expansion cannot lower the stop; pre-entry closes cannot raise it.
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    ind["atr"] = 1.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
