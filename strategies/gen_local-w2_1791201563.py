from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    pattern_window: int = 18
    complexity_floor: float = 0.86
    stretch_atr: float = 0.9
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8
    initial_stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local-w2_1791201563"

    @classmethod
    def params_type(cls) -> type[OrdinalParams]:
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(200, params.pattern_window + 8, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if (params.pattern_window < 6 or params.atr_window < 1
                or not 0 < params.complexity_floor <= 1
                or min(params.stretch_atr, params.breakeven_pct,
                       params.trail_atr, params.initial_stop_atr) <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["r_old"] = out["return"].shift(2)
        out["r_mid"] = out["return"].shift(1)
        a, b, c = out["r_old"], out["r_mid"], out["return"]
        valid = out[["r_old", "r_mid", "return"]].notna().all(axis=1)
        # Equal returns are ordered by their timestamps.
        out["pattern"] = ((a > b).astype(int) + 2 * (a > c).astype(int)
                          + 4 * (b > c).astype(int)).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            name = f"p_{code}"
            out[name] = out["pattern"].eq(code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window).mean()
            probability = out[name]
            entropy -= probability * np.log(probability.clip(lower=1e-12))
        out["complexity"] = entropy / np.log(6.0)
        out["complexity_rise"] = out["complexity"] - out["complexity"].shift(5)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["stretch"] = (close - close.shift(3)) / out["atr"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        needed = ["complexity", "complexity_rise", "ma200", "atr", "stretch"]
        ready = indicators[needed].notna().all(axis=1) & indicators["atr"].gt(0)
        active = (ready & indicators["complexity"].ge(params.complexity_floor)
                  & indicators["complexity_rise"].gt(0.04))
        setups = np.select([
            active & data["close"].gt(indicators["ma200"])
            & indicators["stretch"].lt(-params.stretch_atr),
            active & data["close"].lt(indicators["ma200"])
            & indicators["stretch"].gt(params.stretch_atr)
        ], [1, -1], default=0)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        ready_values = ready.to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, peak, stop_profit = 0.0, 0.0, 0.0
        armed = False
        for i in range(len(data)):
            previous_target = decisions[i - 1] if i else 0
            o, h, l, c = prices[i]
            finite = np.isfinite(prices[i]).all() and min(o, h, l, c) > 0
            if previous_target == 0:
                side = 0
            elif side == 0:
                # Yesterday's decision fills at today's open, never a future open.
                if not finite or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                side = int(previous_target)
                entry, peak, age, armed = o, 0.0, 0, False
                stop_profit = -params.initial_stop_atr * atr[i - 1]
            if side:
                age += 1
                if not finite or not ready_values[i]:
                    continue
                adverse = l - entry if side == 1 else entry - h
                # Test the existing stop before using this bar's favorable extreme.
                if adverse <= stop_profit:
                    continue
                favorable = h - entry if side == 1 else entry - l
                peak = max(peak, favorable)
                armed = armed or peak >= entry * params.breakeven_pct
                if armed:
                    # Profit floor only moves up; short stop prices move down.
                    stop_profit = max(stop_profit, 0.0, peak - params.trail_atr * atr[i])
                close_profit = side * (c - entry)
                regime_ok = c > ma[i] if side == 1 else c < ma[i]
                if close_profit <= stop_profit or age >= 2 or not regime_ok:
                    continue
                decisions[i] = side
            elif finite and ready_values[i]:
                decisions[i] = int(setups[i])
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        # Stop observations cause liquidation at the following open.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), OrdinalParams(trail_atr=0.5)
    index = pd.date_range("2020-01-01", periods=4)
    for direction in (1, -1):
        data = pd.DataFrame({
            "open": [100.0] * 4,
            "high": [100.1, 101.0 if direction == 1 else 100.2, 100.1, 100.1],
            "low": [99.9, 99.8 if direction == 1 else 99.0, 99.9, 99.9],
            "close": [100.0, 100.2 if direction == 1 else 99.8, 100.0, 100.0],
            "volume": [1000.0] * 4
        }, index=index)
        ind = pd.DataFrame({
            "complexity": [0.95] * 4,
            "complexity_rise": [0.1, 0.0, 0.0, 0.0],
            "ma200": [90.0 if direction == 1 else 110.0] * 4,
            "atr": [1.0] * 4,
            "stretch": [-2.0 * direction] * 4
        }, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, 0, 0]
        data["high"], data["low"], data["close"] = 100.1, 99.9, 100.0
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0]
        assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
