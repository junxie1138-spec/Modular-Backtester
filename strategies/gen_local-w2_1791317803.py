from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDriftParams:
    recent_bars: int = 5
    baseline_bars: int = 40
    change_threshold: float = 1.4
    atr_bars: int = 14
    trail_atr: float = 2.8
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapDriftParams]):
    strategy_id = "gen_local-w2_1791317803"

    @classmethod
    def params_type(cls):
        return GapDriftParams

    def warmup_bars(self, params: GapDriftParams) -> int:
        return int(max(params.baseline_bars + params.recent_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapDriftParams) -> pd.DataFrame:
        if min(params.recent_bars, params.baseline_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if params.hold_bars < 1 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("Holding period and ATR multiplier must be positive")
        if not np.isfinite(params.change_threshold) or params.change_threshold <= 0:
            raise ValueError("Change threshold must be positive")
        previous = data["close"].shift(1)
        ratio = data["open"].where(data["open"] > 0) / previous.where(previous > 0)
        gap = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        recent = gap.rolling(params.recent_bars).mean()
        history = gap.shift(params.recent_bars)
        baseline = history.rolling(params.baseline_bars).mean()
        deviation = history.rolling(params.baseline_bars).std(ddof=0)
        standard_error = deviation * np.sqrt(1.0 / params.recent_bars + 1.0 / params.baseline_bars)
        score = (recent - baseline) / standard_error.where(standard_error > 0)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_bars).mean()
        shifted = (baseline < 0) & (recent > 0) & (score >= params.change_threshold)
        shifted &= np.isfinite(score) & np.isfinite(atr) & (atr > 0)
        confirmed = shifted & shifted.shift(1, fill_value=False)
        entry = confirmed & ~confirmed.shift(1, fill_value=False)
        return pd.DataFrame({
            "gap": gap,
            "recent_gap_mean": recent,
            "baseline_gap_mean": baseline,
            "gap_standard_error": standard_error,
            "change_score": score,
            "true_range": tr,
            "atr": atr,
            "regime_shift": shifted,
            "confirmed": confirmed,
            "entry": entry,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDriftParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                if not valid:
                    active = False
                else:
                    # Entry fills on the bar after entry_bar; only in-trade closes enter the high-water mark.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                    if closes[i] <= stop or i - entry_bar >= params.hold_bars:
                        active = False
            elif valid and entries[i]:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapDriftParams(trail_atr=2.0, hold_bars=18)
    index = pd.date_range("2020-01-01", periods=8)
    close = pd.Series([100, 100, 100, 104, 101, 100, 100, 100], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
    context = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = pd.DataFrame({"atr": [1, 1, 1, 1, 10, 1, 1, 1], "entry": [False, True, False, False, False, False, False, False]}, index=index)
    signal = strategy.generate_signals(data, indicators, context, params).data["signal"]
    assert signal.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    params.hold_bars = 3
    flat = data.assign(open=100.0, high=101.0, low=99.0, close=100.0)
    indicators["atr"] = 1.0
    signal = strategy.generate_signals(flat, indicators, context, params).data["signal"]
    assert signal.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    index = pd.date_range("2020-01-01", periods=100)
    gap = np.r_[np.tile([-0.002, -0.001], 30), np.tile([0.002, 0.003], 20)]
    close = pd.Series(100.0 * np.exp(np.cumsum(gap)), index=index)
    previous = close.shift(1).fillna(100.0)
    data = pd.DataFrame({"open": previous * np.exp(gap), "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0})
    indicators = strategy.indicators(data, params)
    assert indicators["entry"].any()
    first = np.flatnonzero(indicators["entry"].to_numpy())[0]
    assert indicators["regime_shift"].iloc[first - 1:first + 1].all()
    pd.testing.assert_frame_equal(indicators.iloc[:75], strategy.indicators(data.iloc[:75], params))
    output = strategy.generate_signals(data, indicators, context, params).data
    assert output["signal"].iloc[first] == 0 and output["signal"].iloc[first + 1] == 1
    assert output["signal"].isin([0, 1]).all() and (output["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
