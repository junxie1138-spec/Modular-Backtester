from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    path_window: int = 12
    drawdown_window: int = 63
    atr_period: int = 14
    area_threshold: float = 0.30
    min_depth: float = 0.035
    min_recovery: float = 0.15
    max_recovery: float = 0.80
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790979881"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.drawdown_window, params.path_window + 1,
                       params.atr_period + 1, 4))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if (min(params.path_window, params.drawdown_window, params.atr_period) < 2
                or params.max_hold < 1 or params.trail_atr <= 0
                or not 0 < params.area_threshold < 1
                or not 0 < params.min_depth < 1
                or not 0 <= params.min_recovery < params.max_recovery <= 1):
            raise ValueError("Invalid recovery parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        out["down"] = (-out["log_return"]).clip(lower=0)
        out["up"] = out["log_return"].clip(lower=0)
        n = params.path_window
        out["down_sum"] = out["down"].rolling(n, min_periods=n).sum()
        out["up_sum"] = out["up"].rolling(n, min_periods=n).sum()
        # Antisymmetric level-two signature of cumulative loss and repair.
        # Each pair contributes positively when the loss precedes the repair.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, n):
            pairs = (out["down"].shift(lag) * out["up"]
                     - out["up"].shift(lag) * out["down"])
            area = area + pairs.rolling(n - lag, min_periods=n - lag).sum()
        out["raw_area"] = area
        denominator = out["down_sum"] * out["up_sum"]
        out["area"] = (area / denominator.where(denominator > 0)).clip(-1, 1)
        out["peak"] = close.rolling(params.drawdown_window,
                                    min_periods=params.drawdown_window).max()
        out["trough"] = close.rolling(params.drawdown_window,
                                      min_periods=params.drawdown_window).min()
        span = out["peak"] - out["trough"]
        out["depth"] = span / out["peak"]
        out["recovery"] = (close - out["trough"]) / span.where(span > 0)
        out["momentum"] = close / close.shift(3) - 1.0
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period).mean()
        out["strength"] = (
            (out["area"] - params.area_threshold) / (1.0 - params.area_threshold)
        ).clip(0, 1).fillna(0.0)
        ready = np.isfinite(out[["area", "depth", "recovery", "momentum", "atr"]]).all(axis=1)
        out["eligible"] = (
            ready & (out["atr"] > 0)
            & (out["area"] > params.area_threshold)
            & (out["depth"] >= params.min_depth)
            & out["recovery"].between(params.min_recovery, params.max_recovery)
            & (out["momentum"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        exposure = 1.0
        for i in range(len(data)):
            if active:
                # The entry decision was yesterday; begin tracking held closes now.
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                else:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if closes[i] <= stop or i - entry_bar >= params.max_hold:
                        active = False
                if not active:
                    continue
            elif eligible[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
                exposure = float(np.clip(0.25 + 0.70 * strength[i], 0.25, 0.95))
            if active:
                signals[i] = 1
                sizes[i] = exposure
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RecoveryParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100, 103, 105, 104, 102, 101, 100]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 10, 10, 10, 10],
        "eligible": [True, False, False, False, False, False, False],
        "strength": [0.5] * 7,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[index[1:5], "size"], 0.60)
    params.max_hold = 2
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    params = RecoveryParams(path_window=4, drawdown_window=4, atr_period=2)
    close = np.exp(np.array([0.0, -0.1, -0.2, -0.1, 0.0]))
    bars = pd.DataFrame({"open": close, "high": close + 0.01,
                         "low": close - 0.01, "close": close, "volume": 100},
                        index=pd.date_range("2020-01-01", periods=5))
    derived = strategy.indicators(bars, params)
    assert np.isclose(derived["area"].iloc[-1], 1.0)
    assert not derived["eligible"].iloc[:4].any()


if __name__ == "__main__":
    _self_check()
