from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel_bars: int = 20
    probe_bars: int = 10
    minimum_probes: int = 2
    roc_bars: int = 5
    acceleration_lag: int = 3
    acceleration_threshold: float = 0.004
    atr_bars: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 3.0
    max_hold: int = 18
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791268597"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(
            params.channel_bars + params.probe_bars + 1,
            params.roc_bars + params.acceleration_lag + 1,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["upper"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["lower"] = data["low"].rolling(params.channel_bars).min().shift(1)
        valid = out["upper"].notna() & out["lower"].notna()
        upper_probe = ((data["high"] > out["upper"]) & (close <= out["upper"])).astype(float).where(valid)
        lower_probe = ((data["low"] < out["lower"]) & (close >= out["lower"])).astype(float).where(valid)
        out["upper_probe"] = upper_probe
        out["lower_probe"] = lower_probe
        out["upper_backlog"] = upper_probe.shift(1).rolling(params.probe_bars).sum()
        out["lower_backlog"] = lower_probe.shift(1).rolling(params.probe_bars).sum()
        out["roc"] = close / close.shift(params.roc_bars).replace(0.0, np.nan) - 1.0
        out["acceleration"] = out["roc"] - out["roc"].shift(params.acceleration_lag)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        long_entry = (
            (out["roc"] > 0.0)
            & (out["acceleration"] > params.acceleration_threshold)
            & (close > out["upper"])
            & (out["upper_backlog"] >= params.minimum_probes)
        )
        short_entry = (
            (out["roc"] < 0.0)
            & (out["acceleration"] < -params.acceleration_threshold)
            & (close < out["lower"])
            & (out["lower_backlog"] >= params.minimum_probes)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        age = 0
        entry = 0.0
        stop = 0.0
        entry_atr = 0.0
        armed = False
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            if direction == 0:
                if entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                    direction = int(entries[i])
                    age = 0
                    entry_atr = float(atrs[i])
                    armed = False
            else:
                # The pending entry from yesterday fills at today's open.
                if age == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        direction = 0
                        continue
                    entry = float(opens[i])
                    stop = direction * entry - params.trail_atr * entry_atr
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0.0:
                    direction = 0
                else:
                    signed_close = direction * closes[i]
                    if signed_close <= stop or age >= params.max_hold:
                        direction = 0
                    else:
                        if direction * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                            armed = True
                        if armed:
                            # Signed stops rise only: long prices rise, short prices fall.
                            stop = max(stop, direction * entry)
                            if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                                stop = max(stop, signed_close - params.trail_atr * atrs[i])
            raw[i] = direction
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0.0, 2.0, 4.0, 1.0, 1.0])
        data = pd.DataFrame({
            "open": 100.0, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": 1000.0,
        }, index=index)
        derived = pd.DataFrame({"entry": [direction, 0, 0, 0, 0], "atr": 1.0}, index=index)
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert (result["size"] > 0.0).all()
        assert strategy.indicators(data, params)["entry"].eq(0).all()


if __name__ == "__main__":
    _self_check()
