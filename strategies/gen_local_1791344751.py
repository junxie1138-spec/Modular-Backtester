from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_days: int = 2
    correlation_window: int = 40
    atr_days: int = 14
    acceleration_threshold: float = 0.10
    echo_threshold: float = 0.15
    stop_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791344751"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.atr_days + 1,
                       params.roc_days + 3 + params.correlation_window))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.roc_days < 1 or params.atr_days < 1 or params.correlation_window < 4:
            raise ValueError("Lookbacks must be positive; correlation_window must be >= 4.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_days).mean()
        out["roc"] = close / close.shift(params.roc_days) - 1.0
        out["acceleration"] = out["roc"].diff()
        out["acceleration_atr"] = (
            out["acceleration"] / (out["atr"] / close).where(out["atr"] > 0)
        )
        out["lag_one"] = out["acceleration"].shift(1)
        out["lag_two"] = out["acceleration"].shift(2)
        out["correlation_one"] = out["acceleration"].rolling(
            params.correlation_window
        ).corr(out["lag_one"])
        out["correlation_two"] = out["acceleration"].rolling(
            params.correlation_window
        ).corr(out["lag_two"])
        # Primitive 1: material positive ROC acceleration.
        out["impulse"] = out["acceleration_atr"] > params.acceleration_threshold
        # Primitive 2: positive two-session acceleration echo, stronger than lag one.
        out["echo"] = (
            (out["correlation_two"] > params.echo_threshold)
            & (out["correlation_two"] > out["correlation_one"])
            & (out["lag_two"] > 0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_days <= 5:
            raise ValueError("hold_days must be between 3 and 5.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        if (not np.isfinite(params.acceleration_threshold)
                or params.acceleration_threshold < 0
                or not np.isfinite(params.echo_threshold)
                or not 0 <= params.echo_threshold < 1):
            raise ValueError("Invalid entry thresholds.")
        ready = np.isfinite(indicators["atr"]) & (indicators["atr"] > 0)
        enter = (
            indicators["impulse"].fillna(False).astype(bool)
            & indicators["echo"].fillna(False).astype(bool) & ready
        ).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # Observe the next open only when its bar is reached.
                active = True
                pending = False
                entry_bar = i
                stop = opens[i] - params.stop_atr * entry_atr
            if active:
                # Freeze both entry price and decision-bar ATR; never trail.
                invalid = not np.isfinite(stop) or not np.isfinite(closes[i])
                expired = i - entry_bar + 1 >= params.hold_days
                if invalid or closes[i] < stop or expired:
                    active = False
                else:
                    raw[i] = 1
            elif enter[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                entry_atr = atr[i]
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0,
        "close": [100.0, 100.0, 104.0, 99.0, 99.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0, "impulse": [True] + [False] * 8,
        "echo": [True] + [False] * 8,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[3], "close"] = 97.0
    stopped = strategy.generate_signals(data, indicators, ctx, params).data
    assert stopped["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0]
    for column in ("impulse", "echo"):
        blocked = indicators.copy()
        blocked[column] = False
        assert not strategy.generate_signals(data, blocked, ctx, params).data["signal"].any()
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(
            data.iloc[:end], indicators.iloc[:end], ctx, params
        ).data
        assert prefix.equals(stopped.iloc[:end])
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
