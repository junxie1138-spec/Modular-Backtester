from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletPositionParams:
    energy_window: int = 8
    coarse_share: float = 0.55
    channel_window: int = 40
    position_threshold: float = 0.65
    migration_lag: int = 5
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[WaveletPositionParams]):
    strategy_id = "gen_local_1791325425"

    @classmethod
    def params_type(cls):
        return WaveletPositionParams

    def warmup_bars(self, params):
        return int(max(16 + params.energy_window - 1,
                       params.channel_window + params.migration_lag,
                       params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        energies = []
        for scale in (4, 8, 16):
            half = scale // 2
            recent = log_close.rolling(half, min_periods=half).mean()
            detail = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            energy = detail.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            out[f"detail_{scale}"] = detail
            out[f"energy_{scale}"] = energy
            energies.append(energy)
        total = energies[0] + energies[1] + energies[2]
        out["coarse_share"] = energies[2].div(total.where(total > 0))
        floor = data["low"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).min()
        ceiling = data["high"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).max()
        width = ceiling - floor
        out["channel_floor"] = floor
        out["channel_ceiling"] = ceiling
        out["position"] = (close - floor).div(width.where(width > 0))
        out["migration"] = out["position"].diff(params.migration_lag)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        energy_agrees = out["coarse_share"].ge(params.coarse_share)
        out["long_entry"] = (
            energy_agrees & out["detail_16"].gt(0)
            & out["position"].ge(params.position_threshold)
            & out["migration"].gt(0)
        ).fillna(False)
        out["short_entry"] = (
            energy_agrees & out["detail_16"].lt(0)
            & out["position"].le(1.0 - params.position_threshold)
            & out["migration"].lt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        buys = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        sells = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        side = 0
        pending = 0
        entry = 0.0
        entry_bar = 0
        stop = -np.inf
        armed = False
        warmup = self.warmup_bars(params)
        for i in range(n):
            if side == 0 and pending != 0:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    side = pending
                    entry = opens[i]
                    entry_bar = i
                    armed = False
                    stop = side * entry - params.trail_atr * atr[i - 1]
                pending = 0
            valid = (np.isfinite(closes[i]) and closes[i] > 0
                     and np.isfinite(atr[i]) and atr[i] > 0)
            if side != 0:
                expired = i - entry_bar + 1 >= params.max_hold
                breached = np.isfinite(closes[i]) and side * closes[i] <= stop
                if expired or breached:
                    side = 0
                    continue
                if valid:
                    favorable_return = side * (closes[i] / entry - 1.0)
                    if favorable_return >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed-price stop only rises, including for shorts.
                        stop = max(stop, side * entry,
                                   side * closes[i] - params.trail_atr * atr[i])
                decisions[i] = side
            elif i >= warmup - 1 and valid:
                if buys[i] and not sells[i]:
                    pending = 1
                elif sells[i] and not buys[i]:
                    pending = -1
                decisions[i] = pending
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=70)
    data = pd.DataFrame({"open": 100.0, "high": 105.0,
                         "low": 98.0, "close": 100.0,
                         "volume": 1000000.0}, index=index)
    data.loc[index[50], "close"] = 104.0
    data.loc[index[51], "close"] = 99.0
    indicators = pd.DataFrame({"atr": 1.0, "long_entry": False,
                               "short_entry": False}, index=index)
    indicators.loc[index[45], "long_entry"] = True
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, indicators, None, WaveletPositionParams()
    ).data
    assert result["signal"].iloc[45] == 0
    assert result["signal"].iloc[46:52].eq(1).all()
    assert result["signal"].iloc[52:].eq(0).all()
    indicators["long_entry"] = False
    indicators.loc[index[45], "short_entry"] = True
    data.loc[index[50], "close"] = 96.0
    data.loc[index[51], "close"] = 101.0
    result = strategy.generate_signals(
        data, indicators, None, WaveletPositionParams()
    ).data
    assert result["signal"].iloc[46:52].eq(-1).all()
    assert result["signal"].iloc[52:].eq(0).all()


if __name__ == "__main__":
    _self_check()
