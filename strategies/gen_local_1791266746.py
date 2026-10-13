from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilParams:
    short_window: int = 6
    baseline_window: int = 60
    compression_ratio: float = 0.55
    atr_floor: float = 0.85
    max_coil_width: float = 4.0
    profit_target: float = 0.02
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[CoilParams]):
    strategy_id = "gen_local_1791266746"

    @classmethod
    def params_type(cls) -> type[CoilParams]:
        return CoilParams

    def warmup_bars(self, params: CoilParams) -> int:
        return int(max(200, params.short_window + 1, params.baseline_window + 1))

    def indicators(self, data: pd.DataFrame, params: CoilParams) -> pd.DataFrame:
        if not (2 <= params.short_window < params.baseline_window):
            raise ValueError("Require 2 <= short_window < baseline_window")
        if not (0 < params.compression_ratio < 1 and params.atr_floor > 0):
            raise ValueError("Invalid volatility thresholds")
        if params.max_coil_width <= 0 or params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("Invalid envelope or exit parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["short_vol"] = out["return"].rolling(params.short_window).std(ddof=0)
        out["baseline_vol"] = out["return"].rolling(params.baseline_window).std(ddof=0)
        out["vol_ratio"] = out["short_vol"] / out["baseline_vol"].replace(0, np.nan)
        out["short_atr"] = out["true_range"].rolling(params.short_window).mean()
        out["baseline_atr"] = out["true_range"].rolling(params.baseline_window).mean()
        out["atr_ratio"] = out["short_atr"] / out["baseline_atr"].replace(0, np.nan)
        out["coil_high"] = data["high"].rolling(params.short_window).max()
        out["coil_low"] = data["low"].rolling(params.short_window).min()
        out["coil_width"] = (out["coil_high"] - out["coil_low"]) / out["baseline_atr"].replace(0, np.nan)
        out["net_progress"] = close.diff(params.short_window)
        out["ma200"] = close.rolling(200).mean()
        out["qualified"] = (
            (out["vol_ratio"] < params.compression_ratio)
            & (out["atr_ratio"] >= params.atr_floor)
            & (out["coil_width"] <= params.max_coil_width)
            & (out["net_progress"] > 0)
            & (close > out["ma200"])
        ).fillna(False)
        out["entry"] = out["qualified"] & ~out["qualified"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoilParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                # The reference entry is the execution-bar open. Exits are close decisions.
                target_hit = np.isfinite(entry_price) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_expired:
                    active = False
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = int(active)
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 103.0
    indicators = pd.DataFrame({"entry": [True, False, False, True, False, False, False, False, False]}, index=index)
    params = CoilParams(max_hold_bars=3)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 1, 0, 0]
    prefix = GeneratedStrategy.generate_signals(None, data.iloc[:5], indicators.iloc[:5], None, params).data
    assert prefix.equals(result.iloc[:5])
    assert (result["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(None, params) >= 200
    warmup = GeneratedStrategy.indicators(None, data, params)
    assert not warmup["entry"].any()


if __name__ == "__main__":
    _self_check()
