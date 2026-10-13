from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalWaveletParams:
    energy_window: int = 16
    atr_period: int = 14
    min_coarse_share: float = 0.55
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SeasonalWaveletParams]):
    strategy_id = "gen_local_1790940531"

    @classmethod
    def params_type(cls):
        return SeasonalWaveletParams

    def warmup_bars(self, params):
        if params.energy_window < 1 or params.atr_period < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not 0.0 < params.min_coarse_share < 1.0 or params.stop_atr <= 0:
            raise ValueError("Invalid energy threshold or stop multiplier.")
        return int(max(16 + params.energy_window, params.atr_period + 1))

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        for scale in (2, 4, 8, 16):
            half = scale // 2
            recent = out["return"].rolling(half, min_periods=half).sum()
            out[f"haar_{scale}"] = (recent - recent.shift(half)) / np.sqrt(scale)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total = out[[f"energy_{s}" for s in (2, 4, 8, 16)]].sum(axis=1, min_count=4)
        coarse = out["energy_8"] + out["energy_16"]
        out["coarse_share"] = coarse / total.where(total > 0)
        out["drift"] = out["return"].rolling(16, min_periods=16).mean()
        out["rms"] = np.sqrt(out["return"].pow(2).rolling(16, min_periods=16).mean())
        out["strength"] = (
            out["coarse_share"] * (out["drift"] / out["rms"].where(out["rms"] > 0)).clip(0, 1)
        ).clip(0, 1).fillna(0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        month = pd.Series(data.index.year * 12 + data.index.month, index=data.index)
        out["month_key"] = month
        out["month_session"] = month.groupby(month, sort=False).cumcount() + 1
        out["season"] = (data.index.month >= 11) | (data.index.month <= 4)
        ready = out[["coarse_share", "haar_8", "haar_16", "drift", "atr"]].notna().all(axis=1)
        out["entry"] = (
            ready & out["season"] & out["month_session"].le(5)
            & out["coarse_share"].ge(params.min_coarse_share)
            & out["haar_8"].gt(0) & out["haar_16"].gt(0)
            & out["drift"].gt(0) & out["atr"].gt(0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        self.warmup_bars(params)
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        months = indicators["month_key"].to_numpy()
        active = False
        pending = False
        last_month = None
        entry_bar = -1
        stop = np.nan
        entry_size = 0.25
        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar = i
                    # Freeze the next-open entry anchor and decision-bar ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                else:
                    active = False
            if active:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = entry_size
                continue
            if entries[i] and months[i] != last_month and i + 1 < n:
                active = True
                pending = True
                last_month = months[i]
                entry_size = float(0.25 + 0.75 * strength[i])
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-11-02", periods=7)
    data = pd.DataFrame({
        "open": [100., 100., 110., 105., 97., 100., 100.],
        "close": [100., 110., 105., 97., 100., 100., 100.],
        "high": 120., "low": 90., "volume": 1000.,
    }, index=index)
    features = pd.DataFrame({
        "atr": [1., 20., 20., 20., 20., 20., 20.],
        "entry": True, "strength": 0.8, "month_key": 24251,
    }, index=index)
    strategy = GeneratedStrategy()
    params = SeasonalWaveletParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.85)
    params.hold_bars = 2
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
