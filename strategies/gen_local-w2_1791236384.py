from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CycleParams:
    roc_bars: int = 5
    memory_bars: int = 48
    cycle_lag: int = 8
    volatility_bars: int = 20
    atr_bars: int = 14
    acceleration_threshold: float = 0.15
    persistence_min: float = 0.20
    inhibition_min: float = 0.15
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.roc_bars, self.volatility_bars, self.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if self.memory_bars < 3 or self.cycle_lag < 2 or self.holding_bars < 1:
            raise ValueError("Invalid memory, lag, or holding period.")
        if not 0 < self.breakeven_pct < 1 or not self.trail_atr > 0:
            raise ValueError("Invalid exit parameters.")
        if not self.acceleration_threshold > 0:
            raise ValueError("Acceleration threshold must be positive.")
        if not 0 <= self.persistence_min < 1 or not 0 <= self.inhibition_min < 1:
            raise ValueError("Correlation thresholds must be in [0, 1).")


class GeneratedStrategy(BaseStrategy[CycleParams]):
    strategy_id = "gen_local-w2_1791236384"

    @classmethod
    def params_type(cls):
        return CycleParams

    def warmup_bars(self, params: CycleParams) -> int:
        return int(max(
            params.roc_bars + 1 + params.cycle_lag + params.memory_bars + 2,
            params.volatility_bars + 2,
            params.atr_bars + 2,
        ))

    def indicators(self, data: pd.DataFrame, params: CycleParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["volatility"] = out["return"].rolling(params.volatility_bars).std()
        out["scaled_acceleration"] = out["acceleration"] / out["volatility"].replace(0, np.nan)
        a = out["acceleration"]
        out["persistence"] = a.rolling(params.memory_bars).corr(a.shift(1)).shift(1)
        out["inhibition"] = a.rolling(params.memory_bars).corr(a.shift(params.cycle_lag)).shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        cycle = (
            (out["persistence"] > params.persistence_min)
            & (out["inhibition"] < -params.inhibition_min)
            & (out["atr"] > 0)
        )
        long_setup = cycle & (out["scaled_acceleration"] > params.acceleration_threshold) & (out["roc"] < 0)
        short_setup = cycle & (out["scaled_acceleration"] < -params.acceleration_threshold) & (out["roc"] > 0)
        out["setup"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        out["confirmed"] = out["setup"].where(out["setup"].eq(out["setup"].shift(1)), 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CycleParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        pending = False
        armed = False
        entry = stop = entry_atr = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    side = 0
                    continue
                entry = opens[i]
                entry_bar = i
                # Signed prices make both long and short stops ratchet upward.
                stop = side * entry - params.trail_atr * entry_atr
                armed = False
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if side:
                signed_close = side * close[i]
                expired = i - entry_bar + 1 >= params.holding_bars
                # Close-based exits submit a flat target for the following open.
                if not valid or signed_close <= stop or expired:
                    side = 0
                    armed = False
                    continue
                if side * (close[i] / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, signed_close - params.trail_atr * atr[i])
                signals[i] = side
            elif valid and confirmed[i] != 0:
                side = int(confirmed[i])
                entry_atr = atr[i]
                pending = True
                signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = CycleParams()
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        prices = 100.0 + side * np.array([0.0, 0.0, 3.0, 2.0, 0.0, 0.0])
        data = pd.DataFrame({
            "open": 100.0, "high": prices + 1.0, "low": prices - 1.0,
            "close": prices, "volume": 1000.0,
        }, index=index)
        indicators = pd.DataFrame({
            "atr": 1.0, "confirmed": [side, 0, 0, 0, 0, 0],
        }, index=index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
        assert result.data["signal"].tolist() == [0, side, side, side, side, 0]
        assert result.data["size"].gt(0).all()
    index = pd.date_range("2020-01-01", periods=100)
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = GeneratedStrategy.indicators(None, flat, params)
    assert indicators["confirmed"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(None, params) >= 64
    index = pd.date_range("2020-01-01", periods=160)
    prices = 100.0 + 3.0 * np.sin(2.0 * np.pi * np.arange(160) / 16.0)
    cycle = pd.DataFrame({"open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices, "volume": 1000.0}, index=index)
    indicators = GeneratedStrategy.indicators(None, cycle, params)
    active = indicators["confirmed"].ne(0)
    assert active.any()
    assert indicators.loc[active, "setup"].eq(indicators["setup"].shift(1).loc[active]).all()


if __name__ == "__main__":
    _self_check()
