from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepaymentParams:
    lookback: int = 20
    recovery_bars: int = 5
    min_drawdown: float = 0.025
    breadth_threshold: float = 0.6
    min_repayment: float = 0.2
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RepaymentParams]):
    strategy_id = "gen_local_1791219770"

    @classmethod
    def params_type(cls):
        return RepaymentParams

    def warmup_bars(self, params: RepaymentParams) -> int:
        return max(params.lookback, params.recovery_bars) + 1

    def indicators(self, data: pd.DataFrame, params: RepaymentParams) -> pd.DataFrame:
        if not (params.lookback >= params.recovery_bars >= 2 and params.hold_bars >= 1):
            raise ValueError("Require lookback >= recovery_bars >= 2 and hold_bars >= 1")
        if not (0 < params.min_drawdown < 1 and 0.5 < params.breadth_threshold < 1
                and 0 < params.min_repayment < 0.8):
            raise ValueError("Invalid drawdown, breadth, or repayment threshold")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["peak"] = close.rolling(params.lookback, min_periods=params.lookback).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["gain"] = out["return"].clip(lower=0)
        out["loss"] = -out["return"].clip(upper=0)
        out["loss_debt"] = out["loss"].rolling(params.lookback, min_periods=params.lookback).sum()
        out["repayment"] = out["gain"].rolling(params.recovery_bars, min_periods=params.recovery_bars).sum()
        out["gain_energy"] = out["gain"].pow(2).rolling(params.recovery_bars, min_periods=params.recovery_bars).sum()
        out["breadth"] = out["repayment"].pow(2) / out["gain_energy"].replace(0, np.nan) / params.recovery_bars
        out["repaid_fraction"] = out["repayment"] / out["loss_debt"].replace(0, np.nan)
        out["ready"] = out.notna().all(axis=1)
        recovering = (out["ready"] & (out["drawdown"] >= params.min_drawdown)
                      & (out["repaid_fraction"] >= params.min_repayment)
                      & (out["repaid_fraction"] <= 0.8))
        broad = out["breadth"] >= params.breadth_threshold
        narrow = out["breadth"] <= 1.0 - params.breadth_threshold
        out["candidate"] = np.select(
            [recovering & broad & (out["return"] > 0),
             recovering & narrow & (out["return"] < 0)],
            [1, -1], default=0)
        out["conviction"] = out["breadth"].where(out["candidate"] == 1, 1.0 - out["breadth"])
        out["entry_size"] = (0.25 + 0.35 * (out["drawdown"] / params.min_drawdown).clip(0, 2) / 2
                             + 0.35 * out["conviction"].clip(0, 1)).clip(0.25, 0.95)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RepaymentParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.25, dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        position, exit_at, held_size = 0, -1, 0.25
        for i in range(n):
            if position:
                if i == exit_at:
                    position = 0
                else:
                    signal[i], size[i] = position, held_size
            elif candidates[i] != 0:
                position = int(candidates[i])
                held_size = float(np.clip(entry_sizes[i], 0.25, 0.95))
                exit_at = i + params.hold_bars
                signal[i], size[i] = position, held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RepaymentParams()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=30))
    checks = pd.DataFrame({"candidate": 0, "entry_size": 0.5}, index=data.index)
    checks.iloc[2, 0] = 1
    checks.iloc[4, 0] = -1
    checks.iloc[12, 0] = -1
    checks.iloc[12, 1] = 0.8
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=21)
    result = GeneratedStrategy().generate_signals(data, checks, ctx, params).data
    assert result["signal"].tolist() == [0] * 3 + [1] * 8 + [0] * 2 + [-1] * 8 + [0] * 9
    assert (result["size"].iloc[3:11] == 0.5).all()
    assert (result["size"].iloc[13:21] == 0.8).all()
