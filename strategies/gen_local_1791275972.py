from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 3
    regime_bars: int = 5
    history_bars: int = 63
    atr_bars: int = 14
    acceleration_quantile: float = 0.85
    regime_quantile: float = 0.70
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791275972"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.roc_bars + params.history_bars + 2,
                       params.regime_bars + params.history_bars,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.roc_bars, params.regime_bars, params.history_bars,
               params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0.5 < params.acceleration_quantile < 1
                and 0.5 < params.regime_quantile < 1
                and params.breakeven_pct > 0 and params.trail_atr > 0):
            raise ValueError("Invalid percentile or stop parameters.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_threshold"] = (
            out["acceleration"].abs().shift(1).rolling(params.history_bars)
            .quantile(params.acceleration_quantile))
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["body"] = (close - data["open"]).abs()
        out["wick"] = (out["range"] - out["body"]).clip(lower=0)
        out["wick_share"] = (
            out["wick"].rolling(params.regime_bars).sum()
            / out["range"].rolling(params.regime_bars).sum().replace(0, np.nan))
        history = out["wick_share"].shift(1).rolling(params.history_bars)
        out["wick_low"] = history.quantile(1 - params.regime_quantile)
        out["wick_high"] = history.quantile(params.regime_quantile)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        required = ["acceleration", "acceleration_threshold", "wick_share",
                    "wick_low", "wick_high", "atr"]
        ready = indicators[required].notna().all(axis=1)
        extreme = (indicators["acceleration"].abs()
                   > indicators["acceleration_threshold"])
        # Body-dominated sessions transmit acceleration; wick-heavy sessions absorb it.
        regime = np.where(indicators["wick_share"] < indicators["wick_low"], 1,
                          np.where(indicators["wick_share"]
                                   > indicators["wick_high"], -1, 0))
        candidate = (np.sign(indicators["acceleration"].fillna(0)).to_numpy()
                     * regime)
        candidate = np.where(ready & extreme, candidate, 0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decision = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry = 0.0
        best = 0.0
        stop = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if position == 0 and pending:
                # The previous close's decision enters at this bar's open.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    position = pending
                    entry = opens[i]
                    best = 0.0
                    stop = -params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                pending = 0
            if position:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    position = 0
                else:
                    profit = position * (closes[i] - entry)
                    # Stops use closing prices; exits execute at the next open.
                    if profit <= stop or age >= params.hold_bars:
                        position = 0
                    else:
                        best = max(best, profit)
                        if best >= entry * params.breakeven_pct:
                            armed = True
                        if armed:
                            stop = max(stop, 0.0)
                            if np.isfinite(atr[i]) and atr[i] > 0:
                                # Signed profit-space stops only rise, also for shorts.
                                stop = max(stop, best - params.trail_atr * atr[i])
                decision[i] = position
            elif candidate[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(candidate[i])
                decision[i] = pending
        df = pd.DataFrame({"signal": decision, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        close = 100 + side * np.array([0, 1, 2, 0, 0, 0], dtype=float)
        data = pd.DataFrame({"open": 100.0, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": 1000.0}, index=index)
        features = pd.DataFrame(
            {"acceleration": [side, 0, 0, 0, 0, 0],
             "acceleration_threshold": 0.5, "wick_share": 0.1,
             "wick_low": 0.2, "wick_high": 0.8, "atr": 1.0}, index=index)
        ctx = StrategyContext("SPY", "1d", 0)
        result = strategy.generate_signals(data, features, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0, 0]
        assert (result["size"] > 0).all()
        short_data = data.iloc[:3]
        short_result = strategy.generate_signals(
            short_data, features.iloc[:3], ctx, params).data
        assert short_result.equals(result.iloc[:3])
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 0.0},
                        index=pd.date_range("2021-01-01", periods=100))
    features = strategy.indicators(flat, Params())
    result = strategy.generate_signals(flat, features, ctx, Params()).data
    assert result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
