from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    atr_window: int = 14
    memory_window: int = 24
    correlation_threshold: float = 0.15
    pulse_margin: float = 0.15
    impulse_min: float = 0.35
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791000834"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(2, params.atr_window) + max(3, params.memory_window) + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(max(2, params.atr_window)).mean()
        out["pulse"] = out["true_range"] / out["atr"].shift(1).replace(0.0, np.nan)
        out["range_autocorrelation"] = out["pulse"].rolling(
            max(3, params.memory_window)
        ).corr(out["pulse"].shift(1)).clip(-1.0, 1.0)
        out["impulse"] = data["close"].diff(2) / out["atr"].replace(0.0, np.nan)
        ready = np.isfinite(out[["atr", "pulse", "range_autocorrelation", "impulse"]]).all(axis=1)
        ready &= out["atr"].gt(0.0)
        substantial = out["impulse"].abs().ge(params.impulse_min)
        continuation = ready & substantial & out["range_autocorrelation"].ge(params.correlation_threshold)
        continuation &= out["pulse"].ge(1.0 + params.pulse_margin)
        reversal = ready & substantial & out["range_autocorrelation"].le(-params.correlation_threshold)
        reversal &= out["pulse"].le(1.0 - params.pulse_margin)
        direction = np.sign(out["impulse"].fillna(0.0))
        out["candidate"] = np.select([continuation, reversal], [direction, -direction], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].ne(0) & out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        side = 0
        entry = np.nan
        entry_bar = -1
        entry_atr = np.nan
        stop = -np.inf
        armed = False
        holding_bars = min(5, max(3, int(params.holding_bars)))
        for i in range(len(data)):
            if side != 0:
                # Yesterday's decision enters at today's open, before today's close is observed.
                if not np.isfinite(entry):
                    entry = opens[i]
                    entry_bar = i
                    stop = side * entry - params.trail_atr * entry_atr
                valid = np.isfinite([entry, closes[i], highs[i], lows[i], atr[i]]).all()
                if not valid:
                    side = 0
                    entry = np.nan
                    continue
                # Stops generate next-open exits; they never assume an intraday stop fill.
                adverse = lows[i] if side == 1 else -highs[i]
                stop_hit = adverse <= stop
                expired = i - entry_bar + 1 >= holding_bars
                if stop_hit or expired:
                    side = 0
                    entry = np.nan
                    armed = False
                else:
                    favorable_return = side * (closes[i] / entry - 1.0)
                    if favorable_return >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed prices make the stop increase monotonically for both directions.
                        stop = max(stop, side * entry, side * closes[i] - params.trail_atr * atr[i])
                signals[i] = side
                continue
            if confirmed[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                side = int(confirmed[i])
                entry = np.nan
                entry_atr = atr[i]
                armed = False
                signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    prices = np.array([100.0, 100.0, 102.0, 99.0, 100.0])
    sample = pd.DataFrame({"open": 100.0, "high": prices + 0.2,
                           "low": prices - 0.2, "close": prices,
                           "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "confirmed": [0, 1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy().generate_signals(sample, derived, None, StrategyParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0]
    assert result.data["size"].gt(0.0).all()
