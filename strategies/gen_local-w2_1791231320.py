from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 63
    burden_window: int = 10
    rank_window: int = 126
    entry_rank: float = 0.80
    min_drawdown: float = 0.02
    profit_target: float = 0.025
    max_hold: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791231320"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(params.peak_window + params.burden_window + params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if min(params.peak_window, params.burden_window, params.rank_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not (0 < params.entry_rank <= 1 and 0 < params.min_drawdown < 1):
            raise ValueError("Invalid entry thresholds")
        if not (params.profit_target > 0 and params.max_hold >= 1):
            raise ValueError("Invalid exit parameters")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["peak"] = close.rolling(params.peak_window).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0)
        out["burden"] = out["depth"].rolling(params.burden_window).sum()
        out["burden_rank"] = out["burden"].rolling(params.rank_window).rank(pct=True)
        out["burden_change"] = out["burden"].diff()
        out["previous_change"] = out["burden_change"].shift(1)
        out["worst_depth"] = out["depth"].rolling(params.burden_window).max()
        out["repair"] = (
            (out["worst_depth"] - out["depth"]) / out["worst_depth"].replace(0, np.nan)
        ).clip(0, 1)
        out["close_change"] = close.diff()
        out["strength"] = (out["burden_rank"] * out["repair"]).fillna(0).clip(0, 1)
        ready = out[["burden_rank", "previous_change", "repair", "close_change"]].notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["burden_rank"] >= params.entry_rank)
            & (out["burden_change"] < 0)
            & (out["previous_change"] >= 0)
            & (out["depth"] >= params.min_drawdown)
            & (out["repair"] > 0)
            & (out["close_change"] > 0)
        ).fillna(False)
        out["entry"] &= np.arange(len(data)) >= self.warmup_bars(params)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownParams,
    ) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, params.min_size, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        held_size = params.min_size

        for i in range(n):
            if active:
                # Read the fill-bar open only once that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                reached_target = (
                    valid_entry
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= params.max_hold
                if not valid_entry or reached_target or timed_out:
                    active = False
                else:
                    raw_signal[i] = 1
                    raw_size[i] = held_size
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                held_size = params.min_size + (params.max_size - params.min_size) * strength[i]
                raw_signal[i] = 1
                raw_size[i] = held_size

        df = pd.DataFrame({"signal": raw_signal, "size": raw_size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"entry": [True, False, False, False, True, False, False],
         "strength": [0.2, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]},
        index=index,
    )
    strategy = GeneratedStrategy()
    params = DrawdownParams(max_hold=2, profit_target=0.02)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1]
    assert np.isclose(result["size"].iloc[1], 0.39)
    assert np.isclose(result["size"].iloc[5], 0.81)
    data.loc[index[1], "close"] = 103.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 1, 1]
    assert (result["size"] > 0).all()
    assert strategy.indicators(data, params)["entry"].sum() == 0


if __name__ == "__main__":
    _self_check()
