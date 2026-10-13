from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalChangeParams:
    diff_order: float = 0.45
    diff_window: int = 32
    recent_window: int = 4
    baseline_window: int = 40
    change_threshold: float = 1.5
    atr_window: int = 14
    breakeven_pct: float = 0.01
    initial_stop_atr: float = 2.0
    trail_atr: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[FractionalChangeParams]):
    strategy_id = "gen_local_1790917628"

    @classmethod
    def params_type(cls) -> type[FractionalChangeParams]:
        return FractionalChangeParams

    def warmup_bars(self, params: FractionalChangeParams) -> int:
        return int(max(200, params.diff_window + params.recent_window + params.baseline_window - 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: FractionalChangeParams) -> pd.DataFrame:
        if not (0.0 < params.diff_order < 1.0):
            raise ValueError("diff_order must be between zero and one")
        if min(params.diff_window, params.recent_window, params.baseline_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two")
        if not (3 <= params.holding_bars <= 5):
            raise ValueError("holding_bars must be between three and five")
        if min(params.change_threshold, params.breakeven_pct, params.initial_stop_atr, params.trail_atr) <= 0:
            raise ValueError("Thresholds and stop distances must be positive")

        close = data["close"].astype(float)
        fractional_close = close.copy()
        weight = 1.0
        for lag in range(1, params.diff_window):
            weight *= (lag - 1 - params.diff_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)

        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["recent_mean"] = fractional_close.rolling(params.recent_window).mean()
        historical = fractional_close.shift(params.recent_window)
        out["baseline_mean"] = historical.rolling(params.baseline_window).mean()
        out["baseline_std"] = historical.rolling(params.baseline_window).std(ddof=0)
        out["change_score"] = (out["recent_mean"] - out["baseline_mean"]) / out["baseline_std"].replace(0.0, np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        out["regime"] = close.gt(out["ma200"])
        out["ready"] = np.isfinite(out[["change_score", "atr", "ma200"]]).all(axis=1) & out["atr"].gt(0)
        out["entry"] = (
            out["ready"]
            & out["regime"]
            & out["change_score"].gt(params.change_threshold)
            & out["change_score"].shift(1).le(params.change_threshold)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FractionalChangeParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        regime = indicators["regime"].fillna(False).to_numpy(dtype=bool)
        targets = np.zeros(len(data), dtype=int)
        active = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        entry_bar = -1

        for i in range(len(data)):
            # Yesterday's target enters at today's open; today's decision fills tomorrow.
            if not active and i > 0 and targets[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    active = True
                    armed = False
                    entry_price = opens[i]
                    entry_bar = i
                    stop = entry_price - params.initial_stop_atr * atrs[i - 1]

            if active:
                if not ready[i] or not np.isfinite(closes[i]):
                    active = False
                    continue
                # Close-observed exits submit a flat target for the next open.
                exit_now = closes[i] <= stop or not regime[i] or i - entry_bar + 1 >= params.holding_bars
                if exit_now:
                    active = False
                    continue
                if closes[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, closes[i] - params.trail_atr * atrs[i])
                targets[i] = 1
            elif entries[i] and ready[i]:
                targets[i] = 1

        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": [100.0, 100.0, 102.0, 101.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "ready": True, "regime": True, "entry": [True, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, FractionalChangeParams(holding_bars=5)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(FractionalChangeParams()) >= 200


if __name__ == "__main__":
    _self_check()
