from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShiftParams:
    fast_bars: int = 4
    baseline_bars: int = 30
    percentile_bars: int = 126
    entry_quantile: float = 0.85
    atr_bars: int = 14
    stop_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShiftParams]):
    strategy_id = "gen_local_1791204444"

    @classmethod
    def params_type(cls) -> type[ShiftParams]:
        return ShiftParams

    def warmup_bars(self, params: ShiftParams) -> int:
        return int(max(params.fast_bars + params.baseline_bars + params.percentile_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ShiftParams) -> pd.DataFrame:
        if min(params.fast_bars, params.baseline_bars, params.percentile_bars, params.atr_bars, params.hold_bars) < 2:
            raise ValueError("Lookbacks and holding period must be at least two bars")
        if not 0.0 < params.entry_quantile < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("Invalid entry quantile or stop multiplier")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        volume = data["volume"].where(data["volume"] > 0.0)
        out["return"] = close.pct_change(fill_method=None)
        out["weighted_return"] = out["return"] * volume
        out["fast_volume_sum"] = volume.rolling(params.fast_bars).sum()
        out["baseline_volume_sum"] = volume.rolling(params.baseline_bars).sum().shift(params.fast_bars)
        out["fast_mean"] = out["weighted_return"].rolling(params.fast_bars).sum() / out["fast_volume_sum"].replace(0.0, np.nan)
        out["baseline_mean"] = out["weighted_return"].rolling(params.baseline_bars).sum().shift(params.fast_bars) / out["baseline_volume_sum"].replace(0.0, np.nan)
        out["baseline_noise"] = out["return"].rolling(params.baseline_bars).std(ddof=0).shift(params.fast_bars)
        out["shift_score"] = (out["fast_mean"] - out["baseline_mean"]) / out["baseline_noise"].replace(0.0, np.nan)
        out["shift_threshold"] = out["shift_score"].rolling(params.percentile_bars).quantile(params.entry_quantile).shift(1)
        out["participation"] = (out["fast_volume_sum"] / params.fast_bars) / (out["baseline_volume_sum"] / params.baseline_bars).replace(0.0, np.nan)
        out["participation_threshold"] = out["participation"].rolling(params.percentile_bars).quantile(0.65).shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        crossing = (out["shift_score"] > out["shift_threshold"]) & (out["shift_score"].shift(1) <= out["shift_threshold"].shift(1))
        out["entry"] = (crossing & (out["fast_mean"] > 0.0) & (out["participation"] > out["participation_threshold"]) & (out["atr"] > 0.0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShiftParams) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop_price = np.nan
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0.0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0.0:
                    active = False
                    continue
                # Freeze the stop using this execution bar's open and the decision bar's ATR.
                stop_price = opens[i] - params.stop_atr * atr[i - 1]
                age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop_price or age >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    sample = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 92.0, "close": [100.0, 100.0, 101.0, 93.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    forced = pd.DataFrame({"entry": [True, False, False, False, False, False, False], "atr": 2.0}, index=index)
    strategy = GeneratedStrategy()
    stopped = strategy.generate_signals(sample, forced, None, ShiftParams()).data
    assert stopped["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    expired = strategy.generate_signals(sample, forced, None, ShiftParams(hold_bars=2)).data
    assert expired["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert stopped["size"].gt(0.0).all()
