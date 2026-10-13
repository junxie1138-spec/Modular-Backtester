from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    ma_window: int = 20
    correlation_window: int = 48
    atr_window: int = 14
    z_entry: float = 0.8
    echo_correlation: float = 0.2
    immediate_correlation_cap: float = 0.15
    forecast_floor: float = 0.001
    stop_atr: float = 1.8
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1790865242"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(params.ma_window, params.correlation_window + 5, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if min(params.ma_window, params.correlation_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 3 <= params.hold_days <= 5 or params.stop_atr <= 0 or params.z_entry <= 0:
            raise ValueError("Require 3-5 holding days and positive stop/z thresholds.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0).replace(0, np.nan)
        out["z"] = ((close - out["ma"]) / out["std"]).replace([np.inf, -np.inf], np.nan)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag4"] = out["return"].shift(4)
        out["lag1"] = out["return"].shift(1)
        out["rho4"] = out["return"].rolling(params.correlation_window).corr(out["lag4"]).clip(-1, 1)
        out["rho1"] = out["return"].rolling(params.correlation_window).corr(out["lag1"]).clip(-1, 1)
        out["recent_return"] = out["return"].rolling(4).sum()
        out["forecast"] = out["rho4"] * out["recent_return"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry"] = (
            (out["z"] <= -params.z_entry)
            & (out["rho4"] >= params.echo_correlation)
            & (out["rho1"].abs() <= params.immediate_correlation_cap)
            & (out["forecast"] >= params.forecast_floor)
            & (out["atr"] > 0)
        ).fillna(False)
        out["strength"] = (
            (-out["z"] / (2 * params.z_entry)) * (out["forecast"] / 0.008)
        ).replace([np.inf, -np.inf], np.nan).clip(0.2, 0.95).fillna(0.2)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: EchoParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.2, dtype=float)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        exposure = 0.2
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Anchor once, when the prior close's order reaches this open.
                    stop = opens[i] - params.stop_atr * entry_atr
                if closes[i] < stop or i - entry_bar + 1 >= params.hold_days:
                    active = False
                else:
                    signal[i] = 1
                    size[i] = exposure
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(strength[i]):
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                exposure = float(np.clip(strength[i], 0.2, 0.95))
                signal[i] = 1
                size[i] = exposure
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = EchoParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 111.0, "low": 94.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame(
        {"entry": [True, False, False, False, False, False, False],
         "atr": 2.0, "strength": 0.7}, index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[result["signal"] == 1, "size"], 0.7)
    data.loc[index[1], "close"] = 110.0
    data.loc[index[2], "close"] = 97.0
    data.loc[index[3], "close"] = 95.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert strategy.indicators(data, params)["entry"].sum() == 0
