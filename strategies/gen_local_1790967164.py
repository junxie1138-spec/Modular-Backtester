from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 16
    min_degree: int = 4
    min_imbalance: int = 2
    flow_window: int = 5
    min_flow: float = 0.15
    vol_window: int = 20
    atr_window: int = 14
    target_vol: float = 0.12
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.05
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790967164"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(params.graph_window + 1, params.flow_window,
                       params.vol_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if min(params.graph_window, params.flow_window, params.vol_window,
               params.atr_window, params.hold_bars) < 2:
            raise ValueError("Lookbacks and hold_bars must be at least two")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Exposure bounds must satisfy 0 < min <= max <= 1")
        if params.target_vol <= 0 or params.stop_atr <= 0:
            raise ValueError("Volatility target and stop multiplier must be positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        upper_blocker = pd.Series(-np.inf, index=data.index)
        lower_blocker = pd.Series(np.inf, index=data.index)
        complete = close.notna()

        # An endpoint edge exists only when every intervening close lies
        # strictly below both endpoints; inversion supplies the lower graph.
        for lag in range(1, params.graph_window + 1):
            previous = close.shift(lag)
            valid = previous.notna() & complete
            upper_degree += (valid & (upper_blocker < np.minimum(close, previous))).astype(float)
            lower_degree += (valid & (lower_blocker > np.maximum(close, previous))).astype(float)
            upper_blocker = np.maximum(upper_blocker, previous)
            lower_blocker = np.minimum(lower_blocker, previous)
            complete &= previous.notna()

        out["upper_degree"] = upper_degree.where(complete)
        out["lower_degree"] = lower_degree.where(complete)
        out["degree_imbalance"] = out["upper_degree"] - out["lower_degree"]

        bar_range = data["high"] - data["low"]
        location = ((2 * close - data["high"] - data["low"])
                    / bar_range.where(bar_range > 0)).clip(-1, 1)
        location = location.where(bar_range != 0, 0.0)
        volume = data["volume"].where(data["volume"] >= 0)
        volume_sum = volume.rolling(params.flow_window).sum()
        out["closing_flow"] = ((location * volume).rolling(params.flow_window).sum()
                               / volume_sum.where(volume_sum > 0))

        previous_close = close.shift(1)
        true_range = pd.concat([
            bar_range,
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        returns = close.pct_change(fill_method=None)
        out["realized_vol"] = returns.rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["realized_vol"].where(
            out["realized_vol"] > 0)).clip(params.min_size, params.max_size)

        topology = ((out["upper_degree"] >= params.min_degree)
                    & (out["degree_imbalance"] >= params.min_imbalance))
        pressure = out["closing_flow"] >= params.min_flow
        valid = np.isfinite(out[["upper_degree", "lower_degree", "closing_flow",
                                 "atr", "realized_vol", "target_size"]]).all(axis=1)
        out["entry_ok"] = (topology & pressure & valid
                           & (out["atr"] > 0) & (out["realized_vol"] > 0))
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry_ok"].fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        active = False
        pending_fill = False
        fixed_stop = np.nan
        entry_atr = np.nan
        age = 0

        for i in range(len(data)):
            if active:
                # Yesterday's decision fills at today's open. Freeze the
                # entry ATR and stop price for the entire holding period.
                if pending_fill:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        pending_fill = False
                        continue
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                    pending_fill = False
                age += 1
                if (not np.isfinite(closes[i])
                        or closes[i] < fixed_stop
                        or age >= params.hold_bars):
                    active = False
                    continue
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                active = True
                pending_fill = True
                entry_atr = atrs[i]
                age = 0
            decisions[i] = int(active)

        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        sizes = indicators["target_size"].replace([np.inf, -np.inf], np.nan)
        df["size"] = sizes.clip(params.min_size, params.max_size).shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 94.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    bars.loc[index[2], "close"] = 95.0
    derived = pd.DataFrame({"entry_ok": False, "atr": 1.0,
                            "target_size": 0.5}, index=index)
    derived.loc[index[0], "entry_ok"] = True
    result = GeneratedStrategy.generate_signals(bars, derived, None, VisibilityParams()).data
    assert result["signal"].tolist() == [0, 1, 1] + [0] * 9
    bars["close"] = 100.0
    result = GeneratedStrategy.generate_signals(bars, derived, None, VisibilityParams()).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    assert result["size"].gt(0).all()
