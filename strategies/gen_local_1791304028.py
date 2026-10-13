from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    debt_window: int = 8
    peak_window: int = 20
    roc_window: int = 3
    atr_window: int = 14
    min_drawdown_atr: float = 1.5
    acceleration_ratio: float = 0.15
    stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791304028"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.peak_window, params.debt_window + 1,
                       params.atr_window + 1, params.roc_window + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.debt_window, params.peak_window, params.roc_window,
               params.atr_window, params.hold_bars) < 1:
            raise ValueError("Window lengths and hold_bars must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (params.min_drawdown_atr, params.acceleration_ratio, params.stop_atr)):
            raise ValueError("Thresholds must be finite and positive.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_window, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["loss_debt"] = (-out["return"].clip(upper=0)).rolling(params.debt_window).sum()
        out["peak"] = close.rolling(params.peak_window).max()
        out["ma200"] = close.rolling(200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out.loc[previous.isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["drawdown_atr"] = (out["peak"] - close) / out["atr"].where(out["atr"] > 0)
        out["service_ratio"] = out["acceleration"] / out["loss_debt"].where(out["loss_debt"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & out["service_ratio"].shift(1).notna()
        deep = out["drawdown_atr"] >= params.min_drawdown_atr
        ratio = out["service_ratio"]
        out["long_entry"] = (
            ready & deep & (close > out["ma200"]) & (out["roc"] < 0)
            & (ratio > params.acceleration_ratio)
            & (ratio.shift(1) <= params.acceleration_ratio)
        ).fillna(False)
        out["short_entry"] = (
            ready & deep & (close < out["ma200"]) & (out["roc"] > 0)
            & (ratio < -params.acceleration_ratio)
            & (ratio.shift(1) >= -params.acceleration_ratio)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        trail = -np.inf
        short_cap = np.inf
        entry_atr = np.nan
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and np.isfinite(opens[i])
                    and np.isfinite(atr[i]) and atr[i] > 0):
                position = 0
                continue
            if position:
                # First iteration after entry is the actual next-bar fill.
                if age == 0:
                    high_water = close[i]
                    short_cap = opens[i] + params.stop_atr * entry_atr
                high_water = max(high_water, close[i])
                trail = max(trail, high_water - params.stop_atr * atr[i])
                age += 1
                # The high-water stop harvests short profits; the cap limits adverse rallies.
                if (close[i] <= trail or age >= params.hold_bars
                        or (position == -1 and close[i] >= short_cap)):
                    position = 0
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
                high_water = np.nan
                trail = -np.inf
                short_cap = np.inf
                entry_atr = atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    close = [100.0, 102.0, 103.0, 102.0, 101.0, 100.0]
    data = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
                        "long_entry": [True, False, False, False, False, False],
                        "short_entry": False}, index=index)
    strategy = GeneratedStrategy()
    params = RecoveryParams(hold_bars=5)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    ind["long_entry"] = False
    ind["short_entry"] = [True, False, False, False, False, False]
    ind["atr"] = 10.0
    result = strategy.generate_signals(data, ind, None, RecoveryParams(hold_bars=3)).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
