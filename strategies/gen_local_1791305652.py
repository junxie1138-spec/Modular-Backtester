from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRecoveryParams:
    path_bars: int = 6
    excursion: float = 0.012
    area_threshold: float = 0.25
    profit_target: float = 0.02
    max_hold: int = 7


class GeneratedStrategy(BaseStrategy[SignatureRecoveryParams]):
    strategy_id = "gen_local_1791305652"

    @classmethod
    def params_type(cls):
        return SignatureRecoveryParams

    @staticmethod
    def warmup_bars(params):
        if not 2 <= params.path_bars <= 9:
            raise ValueError("path_bars must be between 2 and 9")
        return int(params.path_bars + 1)

    @staticmethod
    def indicators(data, params):
        GeneratedStrategy.warmup_bars(params)
        n = params.path_bars
        close = data["close"].where(data["close"] > 0)
        returns = np.log(close).diff()
        # Twice the antisymmetric level-two signature of (time, log-price).
        twice_area = sum(
            (2 * j - n + 1) * returns.shift(n - 1 - j)
            for j in range(n)
        )
        variation = returns.abs().rolling(n, min_periods=n).sum()
        previous = close.shift(1)
        peak = previous.rolling(n, min_periods=n).max()
        trough = previous.rolling(n, min_periods=n).min()
        result = pd.DataFrame({
            "return": returns,
            "signature_area": twice_area / ((n - 1) * variation.replace(0, np.nan)),
            "drawdown": previous / peak - 1,
            "runup": previous / trough - 1,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        if not (params.excursion > 0 and 0 < params.area_threshold <= 1
                and params.profit_target > 0 and 5 <= params.max_hold <= 10):
            raise ValueError("Invalid excursion, area, target, or holding period")
        long_entry = (
            (indicators["drawdown"] <= -params.excursion)
            & (indicators["signature_area"] >= params.area_threshold)
            & (indicators["return"] > 0)
        )
        short_entry = (
            (indicators["runup"] >= params.excursion)
            & (indicators["signature_area"] <= -params.area_threshold)
            & (indicators["return"] < 0)
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if position:
                # A decision from yesterday first fills at today's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                held += 1
                gain = position * (closes[i] / entry_price - 1)
                if gain >= params.profit_target or held >= params.max_hold:
                    position = 0
                    entry_price = np.nan
                    held = 0
            elif entries[i]:
                position = int(entries[i])
                entry_price = np.nan
                held = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureRecoveryParams()
    index = pd.date_range("2020-01-01", periods=20)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 97.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(index) and np.isfinite(derived.to_numpy()).all()
    assert strategy.warmup_bars(params) == 7
    derived.iloc[0] = [0.01, 0.6, -0.02, 0.0]
    derived.iloc[8] = [-0.01, -0.6, 0.0, 0.02]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 7 + [0] + [-1] * 7 + [0] * 4
    data.loc[index[2], "close"] = 103.0
    data.loc[index[9], "close"] = 97.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[1] == 1 and result["signal"].iloc[3] == 0
    assert result["signal"].iloc[9] == -1 and result["signal"].iloc[10] == 0


if __name__ == "__main__":
    _self_check()
