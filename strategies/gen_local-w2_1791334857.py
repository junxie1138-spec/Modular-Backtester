from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    trough_window: int = 10
    streak_days: int = 4
    min_drawdown: float = 0.04
    min_recovery: float = 0.20
    max_recovery: float = 0.60
    atr_window: int = 14
    breakeven_gain: float = 0.025
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791334857"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.peak_window + params.trough_window,
                       params.atr_window + 1, params.streak_days + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        low = data["low"]
        previous_close = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close.div(out["peak"].where(out["peak"] > 0))
        out["recent_depth"] = out["drawdown"].shift(1).rolling(
            params.trough_window, min_periods=params.trough_window
        ).max()
        out["recovery"] = 1.0 - out["drawdown"].div(
            out["recent_depth"].where(out["recent_depth"] > 0)
        )
        out["low_change"] = low.diff()
        higher_low = out["low_change"].gt(0) & low.notna() & low.shift(1).notna()
        out["higher_low_streak"] = higher_low.astype(int).groupby(
            (~higher_low).cumsum()
        ).cumsum().clip(upper=params.streak_days)
        out["true_range"] = pd.concat([
            data["high"] - low,
            (data["high"] - previous_close).abs(),
            (low - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["entry"] = (
            close.gt(out["ma200"])
            & out["recent_depth"].ge(params.min_drawdown)
            & out["drawdown"].gt(0)
            & out["recovery"].between(params.min_recovery, params.max_recovery)
            & out["higher_low_streak"].ge(params.streak_days)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        peak_close = 0.0
        stop = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opening[i]) and opening[i] > 0:
                    active = True
                    entry_price = opening[i]
                    peak_close = entry_price
                    stop = -np.inf
                    armed = False
                    age = 0
            if active:
                age += 1
                if not np.isfinite(close[i]) or close[i] <= 0:
                    active = False
                    continue
                peak_close = max(peak_close, close[i])
                if close[i] >= entry_price * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak_close - params.trail_atr * atr[i])
                # Close-confirmed exits execute at the following open.
                if close[i] <= stop or age >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(close[i]) and close[i] > 0:
                signals[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Checks next-open entry, breakeven activation, a rising stop, and next-open exit.
    index = pd.date_range("2020-01-01", periods=6)
    sample = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [101.0, 104.0, 111.0, 108.0, 107.0, 107.0],
        "low": [99.0, 99.0, 102.0, 106.0, 105.0, 105.0],
        "close": [100.0, 103.0, 110.0, 107.0, 106.0, 106.0],
        "volume": [1000.0] * 6,
    }, index=index)
    derived = pd.DataFrame({
        "atr": [1.0] * 6,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, sample, derived, None, RecoveryParams(trail_atr=2.0)
    )
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()
