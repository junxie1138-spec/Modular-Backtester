from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    channel_bars: int = 10
    pressure_bars: int = 3
    atr_bars: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8
    initial_stop_atr: float = 1.2
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local_1790858699"

    @classmethod
    def params_type(cls) -> type[PressureParams]:
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        return int(max(200, params.channel_bars + 1, params.atr_bars + 1, params.pressure_bars + 1))

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        if min(params.channel_bars, params.pressure_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions")
        if not all(np.isfinite(x) and x > 0 for x in (params.breakeven_pct, params.trail_atr, params.initial_stop_atr)):
            raise ValueError("Stop parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["ceiling"] = data["high"].shift(1).rolling(params.channel_bars, min_periods=params.channel_bars).max()
        out["close_location"] = ((data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0, np.nan)).fillna(0.0)
        out["pressure"] = (data["low"].diff().gt(0) & out["close_location"].ge(0.65)).astype(int)
        resets = out["pressure"].eq(0).cumsum()
        out["pressure_streak"] = out["pressure"].groupby(resets).cumsum()
        out["setup"] = (
            out["pressure_streak"].ge(params.pressure_bars)
            & data["close"].gt(out["ceiling"])
            & data["close"].gt(out["ma200"])
            & out["atr"].gt(0)
        ).fillna(False)
        out["entry_event"] = out["setup"] & ~out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PressureParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_event"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending_entry = False
        armed = False
        entry = stop = peak = 0.0
        held = 0

        for i in range(len(data)):
            # Yesterday's decision fills at today's open; today's decisions
            # become executable only through the mandatory final shift.
            if pending_entry:
                pending_entry = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    entry = opens[i]
                    stop = entry - params.initial_stop_atr * atrs[i - 1]
                    peak = entry
                    held = 0
                    armed = False
                    active = True

            if active:
                held += 1
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                if valid_close:
                    peak = max(peak, closes[i])
                    if closes[i] >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atrs[i])
                # Close-observed stops avoid assuming an intrabar price path.
                exit_now = not valid_close or closes[i] <= stop or held >= params.max_hold_bars
                if exit_now:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
                pending_entry = True

        df = pd.DataFrame({"signal": signals, "size": np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    close = 100.0 + np.arange(240) * 0.5
    sample = pd.DataFrame({"open": close - 0.1, "high": close + 0.2, "low": close - 0.3, "close": close, "volume": 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    params = PressureParams()
    derived = strategy.indicators(sample, params)
    result = strategy.generate_signals(sample, derived, None, params).data
    assert strategy.warmup_bars(params) >= 200
    assert result.index.equals(sample.index)
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()
    assert result["signal"].iloc[:200].eq(0).all()
    assert result["signal"].iloc[200:202].eq(1).all()
    assert result["signal"].iloc[202:].eq(0).all()
    fast = PressureParams(breakeven_pct=0.0001, trail_atr=0.1)
    reversal = sample.copy()
    reversal.loc[index[201], "close"] = reversal.loc[index[200], "close"] - 0.04
    fast_derived = strategy.indicators(reversal, fast)
    fast_derived["entry_event"] = False
    fast_derived.loc[index[199], "entry_event"] = True
    fast_derived.loc[index[201], "atr"] = 100.0
    fast_result = strategy.generate_signals(reversal, fast_derived, None, fast).data
    assert fast_result["signal"].iloc[200] == 1
    assert fast_result["signal"].iloc[201] == 0
