from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalRecoveryParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    percentile_window: int = 126
    entry_quantile: float = 0.12
    profit_target: float = 0.015
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[FractionalRecoveryParams]):
    strategy_id = "gen_local_1791238398"

    @classmethod
    def params_type(cls) -> type[FractionalRecoveryParams]:
        return FractionalRecoveryParams

    def warmup_bars(self, params: FractionalRecoveryParams) -> int:
        return int(params.fractional_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: FractionalRecoveryParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must lie between zero and one")
        if params.fractional_window < 2 or params.percentile_window < 2:
            raise ValueError("indicator windows must be at least two")
        if not (0.0 < params.entry_quantile < 0.5):
            raise ValueError("entry_quantile must lie between zero and one half")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        log_close = np.log(close.where(close > 0.0))
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: fixed finite kernel; increase fractional_window for longer memory.
        fractional_close = log_close.copy()
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[k] * log_close.shift(k)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["lower_threshold"] = fractional_close.shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_quantile)
        out["recovery"] = fractional_close.diff()
        out["entry"] = (
            out["fractional_close"].le(out["lower_threshold"])
            & out["recovery"].gt(0.0)
            & out[["fractional_close", "lower_threshold", "recovery"]].notna().all(axis=1)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalRecoveryParams,
    ) -> SignalFrame:
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError("profit_target and max_hold_bars must be positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        pending_entry = False
        holding = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    holding = True
                    entry_price = opens[i]
                    entry_bar = i
            if holding:
                age = i - entry_bar + 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or age >= params.max_hold_bars:
                    holding = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                signal[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    sample = pd.DataFrame({
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": [100.0, 100.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "entry": [True, False, False, False, True, False, False, False, False]
    }, index=index)
    params = FractionalRecoveryParams(max_hold_bars=3)
    result = GeneratedStrategy.generate_signals(None, sample, derived, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0]
    assert result.data["size"].gt(0.0).all()
