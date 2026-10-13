from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockDecayParams:
    target_vol: float = 0.12
    activation_pct: float = 0.02


class GeneratedStrategy(BaseStrategy[ShockDecayParams]):
    strategy_id = "gen_local_1791231915"

    @classmethod
    def params_type(cls):
        return ShockDecayParams

    def warmup_bars(self, params: ShockDecayParams) -> int:
        return 23

    def indicators(self, data: pd.DataFrame, params: ShockDecayParams) -> pd.DataFrame:
        if not (np.isfinite(params.target_vol) and params.target_vol > 0):
            raise ValueError("target_vol must be finite and positive")
        if not (np.isfinite(params.activation_pct) and 0 < params.activation_pct < 1):
            raise ValueError("activation_pct must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["returns"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        r = out["returns"]
        out["up_energy"] = r.clip(lower=0).pow(2).rolling(21, min_periods=21).sum()
        out["down_energy"] = r.clip(upper=0).pow(2).rolling(21, min_periods=21).sum()
        total = out["up_energy"] + out["down_energy"]
        out["imbalance"] = (out["up_energy"] - out["down_energy"]) / total.where(total > 0)
        out["fast_vol"] = r.rolling(5, min_periods=5).std(ddof=0)
        out["slow_vol"] = r.rolling(21, min_periods=21).std(ddof=0)
        out["drift"] = r.rolling(5, min_periods=5).sum()
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["annual_vol"] = out["slow_vol"] * np.sqrt(252.0)
        out["size"] = (params.target_vol / out["annual_vol"].where(out["annual_vol"] > 0)).clip(0.05, 0.95).fillna(0.05)
        out["cooling_cross"] = (
            (out["fast_vol"] < out["slow_vol"])
            & (out["fast_vol"].shift(1) >= out["slow_vol"].shift(1))
        )
        long_entry = out["cooling_cross"] & (out["imbalance"] > 0.25) & (out["drift"] > 0)
        short_entry = out["cooling_cross"] & (out["imbalance"] < -0.25) & (out["drift"] < 0)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShockDecayParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        pending_entry = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        age = 0
        armed = False
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if pending_entry:
                entry = float(data["open"].iloc[i])
                pending_entry = False
                age = 0
                peak = 0.0
                armed = False
                if not (np.isfinite(entry) and entry > 0):
                    side = 0
            if side:
                age += 1
                if np.isfinite(close) and np.isfinite(atr) and atr > 0:
                    # Signed price distance: stop only increases; a short's price stop decreases.
                    # Stops are evaluated at closes and executed on the following bar.
                    profit = side * (close - entry)
                    peak = max(peak, profit)
                    if peak >= params.activation_pct * entry:
                        armed = True
                    if armed:
                        stop = max(stop, 0.0, peak - 2.5 * atr)
                    if profit <= stop:
                        side = 0
                if age >= 18:
                    side = 0
            else:
                candidate = int(indicators["candidate"].iloc[i])
                if candidate and np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0:
                    side = candidate
                    pending_entry = True
                    stop = -2.5 * atr
            decisions[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = ShockDecayParams()
    for direction in (1, -1):
        index = pd.date_range("2020-01-01", periods=5)
        close = np.array([100.0, 100.0, 100.0 + direction * 3.0, 100.0, 100.0])
        data = pd.DataFrame({"open": 100.0, "high": close + 1.0,
                             "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"atr": 1.0, "size": 0.5,
                                  "candidate": [direction, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=80)
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert result["signal"].eq(0).all()
    assert np.isfinite(result["size"]).all()


if __name__ == "__main__":
    _self_check()
