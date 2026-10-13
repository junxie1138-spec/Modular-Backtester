from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_days: int = 5
    accel_days: int = 3
    budget_days: int = 12
    entry_share: float = 0.65
    atr_days: int = 14
    activation_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold: int = 18
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791309878"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(201, params.roc_days + params.accel_days + params.budget_days + 1, params.atr_days + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.roc_days, params.accel_days, params.budget_days, params.atr_days, params.max_hold) < 1:
            raise ValueError("Lookbacks and max_hold must be positive.")
        if not (0.5 < params.entry_share < 1.0 and params.activation_pct > 0 and params.trail_atr > 0 and 0 < params.size <= 1):
            raise ValueError("Invalid threshold, stop, or size parameter.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close / close.shift(params.roc_days) - 1.0
        out["acceleration"] = out["roc"].diff(params.accel_days)
        out["positive_mass"] = out["acceleration"].clip(lower=0).rolling(params.budget_days, min_periods=params.budget_days).sum()
        out["negative_mass"] = (-out["acceleration"].clip(upper=0)).rolling(params.budget_days, min_periods=params.budget_days).sum()
        out["positive_share"] = out["positive_mass"] / (out["positive_mass"] + out["negative_mass"]).replace(0, np.nan)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - close.shift(1)).abs(), (data["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_days, min_periods=params.atr_days).mean()
        out["regime"] = np.select([close > out["ma200"], close < out["ma200"]], [1, -1], default=0)
        out["aligned_share"] = out["positive_share"].where(out["regime"] == 1, 1.0 - out["positive_share"])
        valid = np.isfinite(out[["ma200", "roc", "acceleration", "aligned_share", "atr"]]).all(axis=1) & (out["atr"] > 0)
        trigger = valid & (out["regime"] != 0) & (out["regime"] == out["regime"].shift(1)) & (out["aligned_share"] >= params.entry_share) & (out["aligned_share"].shift(1) < params.entry_share) & (out["roc"] * out["regime"] < 0) & (out["acceleration"] * out["regime"] > 0)
        out["entry"] = out["regime"].where(trigger, 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        regimes = indicators["regime"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        entry_bar = 0
        armed = False
        stop = 0.0
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                side = 0
                entry = np.nan
                continue
            if side:
                if np.isnan(entry):
                    # The previous close's decision fills at this bar's open.
                    if not (np.isfinite(opens[i]) and opens[i] > 0):
                        side = 0
                        continue
                    entry = opens[i]
                    entry_bar = i
                    armed = False
                    stop = -params.trail_atr * atr[i - 1]
                profit = side * (close[i] - entry)
                # Close-confirmed exits execute at the following open.
                if profit <= stop or regimes[i] != side or i - entry_bar + 1 >= params.max_hold:
                    side = 0
                    entry = np.nan
                    continue
                if profit >= params.activation_pct * entry:
                    armed = True
                if armed:
                    # Signed-profit stop only rises; short stop prices fall.
                    stop = max(stop, 0.0, profit - params.trail_atr * atr[i])
                raw[i] = side
            elif entries[i]:
                side = int(entries[i])
                entry = np.nan
                raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    idx = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=201)
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0, 0, 3, 4, 1, 0, 0])
        data = pd.DataFrame({"open": 100.0, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1000.0}, index=idx)
        ind = pd.DataFrame({"atr": 1.0, "regime": direction, "entry": [0, direction, 0, 0, 0, 0, 0]}, index=idx)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0, 0]
        assert (result["size"] > 0).all()
        prefix = strategy.generate_signals(data.iloc[:4], ind.iloc[:4], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:4])
    assert strategy.warmup_bars(params) >= 200
    idx = pd.date_range("2020-01-01", periods=280)
    t = np.arange(len(idx))
    prices = 100.0 + 0.04 * t + 3.0 * np.sin(t / 9.0)
    data = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1000.0}, index=idx)
    ind = strategy.indicators(data, params)
    assert (ind["entry"].iloc[:200] == 0).all()
    pd.testing.assert_frame_equal(strategy.indicators(data.iloc[:250], params), ind.iloc[:250])
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert set(result["signal"].unique()) <= {-1, 0, 1}
