from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    burden_window: int = 10
    roc_window: int = 4
    acceleration_lag: int = 3
    percentile_window: int = 252
    burden_percentile: float = 0.75
    acceleration_percentile: float = 0.85
    profit_target: float = 0.06
    time_stop: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791261176"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(
            params.peak_window + params.burden_window + params.percentile_window,
            params.roc_window + params.acceleration_lag + params.percentile_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        windows = (params.peak_window, params.burden_window, params.roc_window,
                   params.acceleration_lag, params.percentile_window, params.time_stop)
        if any(value < 1 for value in windows):
            raise ValueError("Lookbacks and time_stop must be positive")
        if not (0 < params.burden_percentile < 1 and
                0 < params.acceleration_percentile < 1):
            raise ValueError("Percentiles must lie between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(params.peak_window).max()
        out["drawdown"] = (1.0 - close / out["peak"]).clip(lower=0)
        out["burden"] = out["drawdown"].rolling(params.burden_window).sum()
        out["roc"] = close.pct_change(params.roc_window, fill_method=None)
        out["acceleration"] = out["roc"].diff(params.acceleration_lag)
        out["burden_threshold"] = out["burden"].rolling(
            params.percentile_window
        ).quantile(params.burden_percentile).shift(1)
        out["acceleration_threshold"] = out["acceleration"].rolling(
            params.percentile_window
        ).quantile(params.acceleration_percentile).shift(1)
        out["drawdown_change"] = out["drawdown"].diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["burden"] > out["burden_threshold"])
            & (out["acceleration"] > out["acceleration_threshold"])
            & (out["acceleration"] > 0)
            & (out["roc"] > 0)
            & (out["drawdown_change"] < 0)
            & (out["drawdown"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if params.time_stop < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Exit parameters must be positive and finite")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = np.nan
        held = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    active = True
                    held = 0
            if active:
                held += 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held >= params.time_stop:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
                pending = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 1000.0}, index=index,
    )
    derived = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    params = RecoveryParams(time_stop=3)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 107.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
