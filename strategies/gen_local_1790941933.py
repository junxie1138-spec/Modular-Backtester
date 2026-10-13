from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCompressionParams:
    compression: float = 0.65
    stop_atr: float = 3.0
    target_vol: float = 0.12
    hold_bars: int = 18
    shock_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GapCompressionParams]):
    strategy_id = "gen_local_1790941933"

    @classmethod
    def params_type(cls):
        return GapCompressionParams

    def warmup_bars(self, params):
        return 200

    def indicators(self, data, params):
        if not (0 < params.compression <= 1 and params.stop_atr > 0
                and params.target_vol > 0 and params.hold_bars >= 1
                and params.shock_atr > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).replace(0, np.nan)
        out["gap"] = data["open"] / previous - 1.0
        out["return"] = data["close"] / previous - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["gap_atr"] = (data["open"] - previous) / out["atr"].shift(1).replace(0, np.nan)
        out["fast_gap_rms"] = out["gap"].pow(2).rolling(5, min_periods=5).mean().pow(0.5)
        out["slow_gap_rms"] = out["gap"].pow(2).rolling(40, min_periods=40).mean().pow(0.5)
        out["volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["shock"] = out["gap_atr"].lt(-params.shock_atr)
        out["recent_shock"] = out["shock"].shift(3).rolling(18, min_periods=18).max().fillna(0).astype(bool)
        required = ["atr", "gap_atr", "fast_gap_rms", "slow_gap_rms", "volatility", "ma200"]
        out["ready"] = np.isfinite(out[required]).all(axis=1) & out["atr"].gt(0) & out["volatility"].gt(0)
        out["entry"] = (
            out["ready"] & out["recent_shock"]
            & data["close"].gt(out["ma200"])
            & out["gap_atr"].between(0.03, 0.50)
            & out["fast_gap_rms"].le(params.compression * out["slow_gap_rms"])
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Entry executes at this bar's open; freeze decision-bar ATR.
                    stop = opens[i] - params.stop_atr * entry_atr
                if (not ready[i] or not np.isfinite(stop)
                        or not np.isfinite(close[i]) or close[i] < stop
                        or close[i] <= regime[i]
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif ready[i] and entry[i]:
                entry_bar = i + 1
                entry_atr = atr[i]
                raw[i] = 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        allocation = (params.target_vol / indicators["volatility"].replace(0, np.nan)).clip(0.01, 0.95)
        df["size"] = allocation.replace([np.inf, -np.inf], np.nan).shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = GapCompressionParams()
    strategy = GeneratedStrategy()
    idx = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=idx)
    ind = pd.DataFrame({"atr": 2.0, "ma200": 90.0, "volatility": 0.20,
                        "ready": True, "entry": False}, index=idx)
    ind.loc[idx[0], "entry"] = True
    ctx = StrategyContext("SPY", "1d", 200)
    result = strategy.generate_signals(data, ind, ctx, p).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 5
    data.loc[idx[2], "close"] = 110.0
    data.loc[idx[3], "close"] = 99.0
    data.loc[idx[4], "close"] = 93.0
    result = strategy.generate_signals(data, ind, ctx, p).data
    assert result["signal"].tolist() == [0] + [1] * 4 + [0] * 19
    assert result["size"].gt(0).all() and np.isfinite(result["size"]).all()
    for n in (1, 4, 5, 18):
        prefix = strategy.generate_signals(data.iloc[:n], ind.iloc[:n], ctx, p).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:n])


if __name__ == "__main__":
    _self_check()
