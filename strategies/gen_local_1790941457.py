from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    level_span: int = 8
    slope_span: int = 5
    reference_window: int = 40
    atr_window: int = 14
    surprise_allowance: float = 0.35
    capacity: float = 4.0
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790941457"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.atr_window + 1,
                   params.level_span + params.slope_span + params.reference_window + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.level_span, params.slope_span, params.atr_window,
               params.hold_bars) < 1 or params.reference_window < 2:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (params.capacity, params.breakeven_pct, params.trail_atr)):
            raise ValueError("Capacity and stop parameters must be finite and positive.")
        if not np.isfinite(params.surprise_allowance) or params.surprise_allowance < 0:
            raise ValueError("Surprise allowance must be finite and nonnegative.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["level"] = out["log_close"].ewm(
            span=params.level_span, adjust=False, min_periods=params.level_span).mean()
        out["level_change"] = out["level"].diff()
        out["slope"] = out["level_change"].ewm(
            span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
        prior = out["slope"].shift(1)
        out["expected_slope"] = prior.rolling(params.reference_window).mean()
        out["slope_noise"] = prior.rolling(params.reference_window).std(ddof=0)
        out["innovation"] = ((out["slope"] - out["expected_slope"]) /
                             out["slope_noise"].where(out["slope_noise"] > 1e-12))
        out["innovation"] = out["innovation"].replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["valid"] = (out["innovation"].notna() & out["slope"].notna() &
                        out["atr"].gt(0) & np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1))
        out["valid"] &= np.arange(len(data)) >= self.warmup_bars(params)
        out["surplus"] = (out["innovation"].clip(-3.0, 3.0) -
                          params.surprise_allowance).where(out["valid"], 0.0)
        out["cumulative_surplus"] = out["surplus"].cumsum()
        out["capacity_load"] = (out["cumulative_surplus"] -
                                out["cumulative_surplus"].cummin().clip(upper=0.0))
        out["candidate"] = (out["valid"] & out["capacity_load"].ge(params.capacity) &
                            out["slope"].gt(0) & out["innovation"].gt(0))
        out["confirmed"] = out["candidate"] & out["candidate"].shift(1, fill_value=False)
        out["entry_event"] = out["confirmed"] & ~out["confirmed"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = peak = 0.0
        entry_bar = 0
        prices = data[["open", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        events = indicators["entry_event"].fillna(False).to_numpy(dtype=bool)
        for i, (opening, low, close) in enumerate(prices):
            exited = False
            if pending:
                pending = False
                if np.isfinite(opening) and opening > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    entry = peak = opening
                    stop = entry - params.trail_atr * atr[i - 1]
                    entry_bar = i
            if active:
                # Test the previously established stop before using this bar's close.
                if (np.isfinite(low) and low <= stop) or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                    exited = True
                else:
                    if np.isfinite(close):
                        peak = max(peak, close)
                        armed = armed or close >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                    raw[i] = 1
            if not active and not exited and events[i]:
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [101.0, 101.0, 101.0, 104.0, 106.0, 106.0, 106.0],
        "low": [99.0, 99.0, 99.0, 99.0, 102.0, 100.5, 104.0],
        "close": [100.0, 100.0, 100.0, 103.0, 105.0, 105.0, 105.0],
        "volume": [1000.0] * 7
    }, index=index)
    ind = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0],
        "entry_event": [False, False, True, False, False, False, False]
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0]
    derived = strategy.indicators(data, params)
    assert not derived["entry_event"].any()
    assert (derived["confirmed"] == (derived["candidate"] &
            derived["candidate"].shift(1, fill_value=False))).all()
