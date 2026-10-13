from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryParams:
    entry_z: float = 0.75
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[ExpiryParams]):
    strategy_id = "gen_local-w2_1791209946"

    @classmethod
    def params_type(cls):
        return ExpiryParams

    def warmup_bars(self, params: ExpiryParams) -> int:
        return 41

    def indicators(self, data: pd.DataFrame, params: ExpiryParams) -> pd.DataFrame:
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        close = data["close"]
        mean = close.rolling(40, min_periods=40).mean()
        std = close.rolling(40, min_periods=40).std(ddof=0).replace(0, np.nan)
        previous_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["ma"] = mean
        out["std"] = std
        out["z"] = ((close - mean) / std).replace([np.inf, -np.inf], np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["tr"] = tr
        out["atr"] = tr.rolling(14, min_periods=14).mean()
        out["third_friday"] = (
            (data.index.dayofweek == 4)
            & (data.index.day >= 15)
            & (data.index.day <= 21)
        )
        out["season_window"] = out["third_friday"].astype(int).rolling(4, min_periods=1).max().eq(1)
        out["entry"] = (
            out["season_window"]
            & out["z"].le(-params.entry_z)
            & out["z"].gt(out["previous_z"])
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ExpiryParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        months = data.index.year * 12 + data.index.month
        active = False
        pending = False
        stop = np.nan
        entry_bar = -1
        used_month = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry_bar = i
                    # Freeze the actual entry-bar open and decision-bar ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar + 1 >= 18:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and months[i] != used_month and np.isfinite(closes[i]):
                signals[i] = 1
                pending = True
                used_month = months[i]
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2024-01-01", periods=24)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame({"atr": 1.0, "entry": False}, index=index)
    ind.iloc[0, ind.columns.get_loc("entry")] = True
    result = GeneratedStrategy.generate_signals(None, data, ind, None, ExpiryParams()).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 5
    data.iloc[1, data.columns.get_loc("open")] = 110.0
    data.iloc[1, data.columns.get_loc("close")] = 108.0
    data.iloc[2, data.columns.get_loc("close")] = 107.0
    ind.iloc[1:, ind.columns.get_loc("atr")] = 0.1
    result = GeneratedStrategy.generate_signals(None, data, ind, None, ExpiryParams()).data
    assert result["signal"].tolist() == [0, 1, 1] + [0] * 21


if __name__ == "__main__":
    _self_check()
