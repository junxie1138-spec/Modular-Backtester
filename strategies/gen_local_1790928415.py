from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeDilationParams:
    migration_bars: int = 3
    vol_lookback: int = 20
    atr_period: int = 14
    strength_threshold: float = 0.25
    target_vol: float = 0.12
    stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeDilationParams]):
    strategy_id = "gen_local_1790928415"

    @classmethod
    def params_type(cls):
        return RangeDilationParams

    def warmup_bars(self, params):
        return int(max(params.migration_bars + 1, params.vol_lookback, params.atr_period + 1))

    def indicators(self, data, params):
        if min(params.migration_bars, params.vol_lookback, params.atr_period) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 0 < params.strength_threshold < 1 or params.target_vol <= 0 or params.stop_atr <= 0:
            raise ValueError("Invalid strength, volatility target, or stop.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("Holding horizon must be three to five sessions.")
        out = pd.DataFrame(index=data.index)
        high = data["high"].where(data["high"] > 0)
        low = data["low"].where(data["low"] > 0)
        out["high_advance"] = high.diff(params.migration_bars)
        out["low_advance"] = low.diff(params.migration_bars)
        total = out["high_advance"].abs() + out["low_advance"].abs()
        out["strength"] = ((out["high_advance"] - out["low_advance"]) / total.where(total > 0)).clip(0, 1)
        out["log_range"] = np.log(high / low).where(high >= low)
        out["range_variance"] = out["log_range"].pow(2) / (4 * np.log(2))
        out["annual_vol"] = np.sqrt(252 * out["range_variance"].rolling(params.vol_lookback).mean())
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
        ).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["high_advance"] > 0)
            & (out["low_advance"] >= 0)
            & (out["strength"] >= params.strength_threshold)
            & (out["annual_vol"] > 0)
            & (out["atr"] > 0)
            & (data["close"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        candidate_size = (
            params.target_vol * indicators["strength"] / indicators["annual_vol"].where(indicators["annual_vol"] > 0)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 1.0).to_numpy()
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        active = False
        stop = 0.0
        entry_atr = 0.0
        held = 0
        exposure = 1.0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    held = 0
                    # Freeze the entry open and the ATR known when the order was decided.
                    stop = opens[i] - params.stop_atr * entry_atr
            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or held >= params.hold_bars:
                    active = False
                    continue
                signal[i] = 1
                size[i] = exposure
            elif entry[i] and i >= self.warmup_bars(params) - 1:
                pending = True
                entry_atr = atr[i]
                exposure = candidate_size[i]
                signal[i] = 1
                size[i] = exposure
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeDilationParams(migration_bars=1, vol_lookback=1, atr_period=1)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 110.0, "high": 112.0, "low": 108.0, "close": 110.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"entry": [False, True, False, False, False, False, False, False], "atr": 2.0, "strength": 0.5, "annual_vol": 0.2}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=2)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[result["signal"] == 1, "size"], 0.3)
    data.loc[index[3], "close"] = 105.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
