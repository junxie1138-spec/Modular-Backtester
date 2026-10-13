from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    path_window: int = 6
    flow_window: int = 3
    atr_window: int = 14
    area_threshold: float = 0.04
    flow_threshold: float = 0.12
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791284121"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.path_window + 1, params.flow_window, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.path_window, params.flow_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        n = params.path_window
        out["path_net"] = out["log_return"].rolling(n).sum()
        out["path_length"] = out["log_return"].abs().rolling(n).sum()
        # Antisymmetric level-two signature of (elapsed time, log price).
        # Positive area means gains concentrate late in the path.
        area = sum((n - 1 - 2 * lag) * out["log_return"].shift(lag) for lag in range(n))
        out["path_area"] = 0.5 * area / (n * out["path_length"].replace(0.0, np.nan))
        out["bar_range"] = (data["high"] - data["low"]).where(lambda s: s >= 0)
        out["body_pressure"] = ((data["close"] - data["open"]) / out["bar_range"].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out.loc[out["bar_range"].eq(0), "body_pressure"] = 0.0
        out["volume_effort"] = out["body_pressure"] * data["volume"].where(data["volume"] >= 0)
        out["volume_sum"] = data["volume"].where(data["volume"] >= 0).rolling(params.flow_window).sum()
        out["flow"] = out["volume_effort"].rolling(params.flow_window).sum() / out["volume_sum"].replace(0.0, np.nan)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([out["bar_range"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between three and five sessions.")
        if min(params.breakeven_pct, params.trail_atr, params.initial_stop_atr) <= 0:
            raise ValueError("Stop parameters must be positive.")
        eligible = ((indicators["path_area"] > params.area_threshold)
                    & (indicators["path_net"] > 0)
                    & (indicators["flow"] > params.flow_threshold)
                    & (indicators["atr"] > 0)).fillna(False).to_numpy()
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and min(prices[i]) > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if pending:
                pending = False
                if valid:
                    # Previous close's instruction fills at this open.
                    entry = peak = opening
                    stop = entry - params.initial_stop_atr * atr[i - 1]
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                if not valid:
                    active = False
                    continue
                # Existing stops can be touched during this bar. Exit orders
                # still execute next bar, as required by the framework.
                touched = low <= stop
                peak = max(peak, high)
                armed = armed or high >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # A newly raised stop is checked against the close; earlier
                # lows cannot establish whether the high occurred first.
                if touched or close <= stop or age >= params.max_hold:
                    active = False
                    continue
                signals[i] = 1
            elif valid and eligible[i]:
                signals[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 100, 101, 103, 102, 102, 102],
        "high": [100, 101.2, 104, 103, 103, 103, 103],
        "low": [99, 99.5, 100.5, 101.8, 101, 101, 101],
        "close": [100, 101, 103, 102, 102, 102, 102],
        "volume": [1000] * 7,
    }, index=index)
    indicators = pd.DataFrame({
        "path_area": [0.2] + [0.0] * 6,
        "path_net": [0.01] * 7,
        "flow": [0.5] * 7,
        "atr": [1.0] * 7,
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(max_hold=5)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    prefix = strategy.generate_signals(data.iloc[:3], indicators.iloc[:3], None, params).data
    assert result.iloc[:3].equals(prefix)
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    calculated = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, calculated, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
