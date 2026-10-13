from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    fractional_order: float = 0.5
    profit_target: float = 0.015


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1791197208"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    @staticmethod
    def warmup_bars(params: WaveParams) -> int:
        return 72

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        close = data["close"].astype(float).where(data["close"] > 0)
        close = close.replace([np.inf, -np.inf], np.nan)
        weights = np.ones(32, dtype=float)
        for k in range(1, 32):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: fixed 32-tap memory; extend the kernel if longer memory is required.
        fractional_close = close.copy()
        for k in range(1, 32):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["fractional_mean"] = fractional_close.rolling(20, min_periods=20).mean()
        out["fractional_std"] = fractional_close.rolling(20, min_periods=20).std(ddof=0)
        out["displacement"] = (
            (fractional_close - out["fractional_mean"])
            / out["fractional_std"].replace(0.0, np.nan)
        )
        out["increment"] = fractional_close.diff()
        out["lagged_increment"] = out["increment"].shift(1)
        out["persistence"] = out["increment"].rolling(20, min_periods=20).corr(out["lagged_increment"])
        out = out.replace([np.inf, -np.inf], np.nan)
        traveling = (
            out["persistence"].gt(0.0)
            & out["displacement"].gt(0.0)
            & out["displacement"].shift(1).le(0.0)
        )
        standing = (
            out["persistence"].le(0.0)
            & out["displacement"].lt(-0.5)
            & out["increment"].gt(0.0)
            & out["lagged_increment"].le(0.0)
        )
        out["entry"] = (traveling | standing).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        fill_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == fill_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_stop = i - fill_bar + 1 >= 4
                if target_hit or time_stop:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                signal[i] = 1
                active = True
                fill_bar = i + 1
                entry_price = np.nan
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    sample = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    sample.loc[index[2], "close"] = 102.0
    triggers = pd.DataFrame({"entry": [True, False, False, False, True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(sample, triggers, None, WaveParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0.0).all()
    derived = GeneratedStrategy.indicators(sample, WaveParams())
    assert not derived["entry"].any()
