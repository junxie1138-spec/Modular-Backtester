from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    box_bars: int = 5
    baseline_bars: int = 20
    compression_ratio: float = 0.75
    loading_ratio: float = 1.05
    breakout_volume: float = 1.20
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791237961"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.box_bars, params.baseline_bars) + 3)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(p.baseline_bars, min_periods=p.baseline_bars).mean()
        out["short_tr"] = out["tr"].rolling(p.box_bars, min_periods=p.box_bars).mean()
        out["volume_base"] = data["volume"].rolling(p.baseline_bars, min_periods=p.baseline_bars).mean()
        out["volume_box"] = data["volume"].rolling(p.box_bars, min_periods=p.box_bars).mean()
        out["ceiling"] = data["high"].rolling(p.box_bars, min_periods=p.box_bars).max().shift(1)
        out["floor"] = data["low"].rolling(p.box_bars, min_periods=p.box_bars).min().shift(1)
        out["compression"] = out["short_tr"].shift(1).div(out["atr"].shift(1).replace(0.0, np.nan))
        out["loading"] = out["volume_box"].shift(1).div(out["volume_base"].shift(1).replace(0.0, np.nan))
        out["relative_volume"] = data["volume"].div(out["volume_base"].shift(1).replace(0.0, np.nan))
        out["box_width_atr"] = (out["ceiling"] - out["floor"]).div(out["atr"].shift(1).replace(0.0, np.nan))
        out["first_break"] = (
            (out["compression"] <= p.compression_ratio)
            & (out["loading"] >= p.loading_ratio)
            & (out["box_width_atr"] <= 3.0)
            & (out["relative_volume"] >= p.breakout_volume)
            & (data["close"] > out["ceiling"])
            & (data["close"] > data["open"])
        ).fillna(False)
        out["entry_ready"] = (
            out["first_break"].shift(1, fill_value=False)
            & (data["close"] > out["ceiling"].shift(1))
            & (data["close"] > out["prior_close"])
            & (data["close"] > data["open"])
            & (out["relative_volume"] >= p.breakout_volume)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        p = params
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        target = np.zeros(len(data), dtype=int)
        holding = False
        entry = np.nan
        stop = np.nan
        armed = False
        age = 0
        for i in range(len(data)):
            if holding:
                # The previous close's entry decision fills at this bar's open.
                if np.isnan(entry):
                    if i == 0 or not np.isfinite(opening[i]) or opening[i] <= 0.0 or not np.isfinite(atr[i - 1]):
                        holding = False
                        continue
                    entry = opening[i]
                    stop = entry - p.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                age += 1
                if np.isfinite(close[i]) and close[i] > 0.0:
                    if close[i] >= entry * (1.0 + p.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0.0:
                            stop = max(stop, close[i] - p.trail_atr * atr[i])
                    # Close-based exits execute at the following open.
                    if close[i] <= stop or age >= p.max_hold_bars:
                        holding = False
                        entry = np.nan
                        continue
                elif age >= p.max_hold_bars:
                    holding = False
                    entry = np.nan
                    continue
                target[i] = 1
            elif ready[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                holding = True
                entry = np.nan
                target[i] = 1
        df = pd.DataFrame({"signal": target, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [101.0, 103.0, 105.0, 102.0, 101.0],
        "low": [99.0, 99.0, 99.0, 99.0, 99.0],
        "close": [100.0, 102.0, 104.0, 101.0, 100.0],
        "volume": [1000.0] * 5,
    }, index=index)
    derived = pd.DataFrame({
        "atr": [1.0] * 5,
        "entry_ready": [False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, CompressionParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0]
    assert (result.data["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
