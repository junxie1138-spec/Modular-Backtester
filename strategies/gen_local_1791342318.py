from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    analog_window: int = 126
    trail_atr: float = 1.5

    def __post_init__(self):
        if not isinstance(self.analog_window, int) or self.analog_window < 20:
            raise ValueError("analog_window must be an integer >= 20")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1791342318"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return params.analog_window + 22

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).replace(0, np.nan)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["range_pct"] = out["true_range"] / previous
        out["volatility"] = out["range_pct"].rolling(3, min_periods=3).mean()
        vol = out["volatility"].where(out["volatility"] > 0)
        out["vol_now"] = np.log(vol / vol.shift(1))
        out["vol_before"] = out["vol_now"].shift(1)
        out["strain"] = (
            data["close"].diff(3)
            / out["true_range"].rolling(3, min_periods=3).sum().replace(0, np.nan)
        ).clip(-1.0, 1.0)
        features = out[["vol_now", "vol_before", "strain"]]
        support = pd.Series(0.0, index=data.index)
        payoff = pd.Series(0.0, index=data.index)
        # All analog outcomes finish before today's shape begins.
        for lag in range(6, params.analog_window + 6):
            distance = (
                (features - features.shift(lag))
                / np.array([0.20, 0.20, 0.25])
            ).pow(2).sum(axis=1, min_count=3)
            outcome = (
                data["open"].shift(lag - 3)
                / data["open"].shift(lag - 1).replace(0, np.nan) - 1.0
            ).replace([np.inf, -np.inf], np.nan)
            weight = np.exp(-0.5 * distance).where(outcome.notna(), 0.0).fillna(0.0)
            support += weight
            payoff += weight * outcome.fillna(0.0)
        out["analog_support"] = support
        out["analog_forecast"] = payoff / support.replace(0, np.nan)
        out["entry_ready"] = (
            (out["vol_now"] < 0)
            & (out["vol_before"] > 0)
            & (out["strain"] < -0.15)
            & (support >= 5.0)
            & (out["analog_forecast"] > 0.002)
            & out["atr"].gt(0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShapeParams
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        held = 0
        for i, (opening, high, low, close) in enumerate(prices):
            if not np.isfinite([opening, high, low, close, atr[i]]).all() or atr[i] <= 0:
                active = pending = False
                continue
            if pending:
                # Entry uses this bar's open only after yesterday's decision.
                entry = opening
                stop = entry - params.trail_atr * atr[i - 1]
                peak = entry
                held = 0
                armed = False
                active = True
                pending = False
            if active:
                held += 1
                # Test the existing stop before using this bar's high.
                breached = low <= stop
                if not breached:
                    peak = max(peak, high)
                    armed = armed or peak >= entry * 1.005
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    breached = close <= stop
                if breached or held >= 2:
                    active = False
                else:
                    decisions[i] = 1
                # A close exit is filled next open; do not re-enter this bar.
                continue
            if ready[i]:
                decisions[i] = 1
                pending = True
        df["signal"] = decisions
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [100.0, 101.0, 101.0, 100.0, 101.0],
        "low": [99.0, 99.5, 99.5, 99.0, 99.5],
        "close": [100.0, 100.8, 100.8, 100.0, 100.8],
        "volume": [1000.0] * 5,
    }, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry_ready": True}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, ShapeParams()).data
    # Newly armed breakeven must not retroactively test the same bar's low.
    assert result["signal"].tolist() == [0, 1, 1, 0, 1]
    data.loc[index[1], "low"] = 98.0
    stopped = strategy.generate_signals(data, ind, None, ShapeParams()).data
    assert stopped["signal"].tolist() == [0, 1, 0, 1, 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
