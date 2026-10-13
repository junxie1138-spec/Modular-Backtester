from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBandParams:
    min_depth: float = 0.03
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[DrawdownBandParams]):
    strategy_id = "gen_local-w2_1790941907"

    @classmethod
    def params_type(cls):
        return DrawdownBandParams

    def warmup_bars(self, params: DrawdownBandParams) -> int:
        return 46

    def indicators(self, data: pd.DataFrame, params: DrawdownBandParams) -> pd.DataFrame:
        if not (np.isfinite(params.min_depth) and 0 < params.min_depth < 1):
            raise ValueError("min_depth must be between zero and one")
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("stop_atr must be positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["peak"] = data["close"].rolling(40, min_periods=40).max()
        out["depth"] = 1.0 - data["close"] / out["peak"].replace(0.0, np.nan)
        out["previous_depth"] = out["depth"].shift(1)
        out["band_min"] = out["previous_depth"].rolling(6, min_periods=6).min()
        out["band_max"] = out["previous_depth"].rolling(6, min_periods=6).max()
        out["band_mean"] = out["previous_depth"].rolling(6, min_periods=6).mean()
        out["compression"] = (
            (out["band_max"] - out["band_min"])
            / out["band_mean"].replace(0.0, np.nan)
        )
        out["previous_peak"] = out["peak"].shift(6)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            out["peak"].eq(out["previous_peak"])
            & out["band_min"].ge(params.min_depth)
            & out["compression"].le(0.25)
            & out["depth"].lt(out["band_min"])
            & out["depth"].gt(0.0)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownBandParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # This open is observed only on the actual fill bar.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                if (
                    not np.isfinite(stop)
                    or not np.isfinite(closes[i])
                    or closes[i] < stop
                    or i - entry_bar + 1 >= 8
                ):
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_bar = i + 1
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({
        "open": [100.0] + [105.0] * 11,
        "high": 113.0, "low": 99.0,
        "close": [100.0, 106.0, 112.0, 104.0, 102.0] + [104.0] * 7,
        "volume": 1000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = DrawdownBandParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=46)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [True] + [False] * 11}, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    data["close"] = 106.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    empty_entries = strategy.indicators(data, params)
    assert not empty_entries["entry"].any()
