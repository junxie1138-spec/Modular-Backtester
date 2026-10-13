from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fd_order: float = 0.8
    fd_window: int = 32
    regime_window: int = 40
    elastic_ceiling: float = 0.25
    entry_z: float = -0.75
    pressure_window: int = 5
    pressure_min: float = 0.10
    profit_target: float = 0.025
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791241828"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(params.fd_window + params.regime_window, params.pressure_window))

    def indicators(self, data, params):
        if not (0 < params.fd_order < 1 and params.fd_window >= 2
                and params.regime_window >= 3 and params.pressure_window >= 1
                and params.profit_target > 0 and params.max_hold >= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.ones(params.fd_window, dtype=float)
        for k in range(1, params.fd_window):
            weights[k] = -weights[k - 1] * (params.fd_order - k + 1) / k
        fd = close * weights[0]
        for k in range(1, params.fd_window):
            fd = fd + weights[k] * close.shift(k)
        out["fd_close"] = fd
        out["fd_lag"] = fd.shift(1)
        roll = fd.rolling(params.regime_window, min_periods=params.regime_window)
        out["fd_mean"] = roll.mean()
        out["fd_std"] = roll.std(ddof=0).replace(0.0, np.nan)
        out["fd_z"] = (fd - out["fd_mean"]) / out["fd_std"]
        out["fd_z_lag"] = out["fd_z"].shift(1)
        out["fd_correlation"] = roll.corr(out["fd_lag"])
        out["elastic_regime"] = out["fd_correlation"].le(params.elastic_ceiling)
        out["range"] = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["closing_pressure"] = (
            (2.0 * close - data["high"] - data["low"]) / out["range"]
        ).clip(-1.0, 1.0).fillna(0.0)
        volume = data["volume"].astype(float).clip(lower=0.0)
        out["signed_volume"] = out["closing_pressure"] * volume
        out["volume_sum"] = volume.rolling(params.pressure_window).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.pressure_window).sum()
        out["volume_pressure"] = (
            out["signed_volume_sum"] / out["volume_sum"].replace(0.0, np.nan)
        )
        out["fd_rebound"] = (
            out["elastic_regime"]
            & out["fd_z"].le(params.entry_z)
            & out["fd_z"].gt(out["fd_z_lag"])
        )
        out["entry_ready"] = (
            out["fd_rebound"] & out["volume_pressure"].gt(params.pressure_min)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                # A previous close's entry decision fills at this bar's open.
                if entry_bar == i:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold
                if target_hit or expired:
                    active = False
                    continue
            elif ready[i] and (i == 0 or not ready[i - 1]):
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=4)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    ready = pd.DataFrame({"entry_ready": [True, False, False, False]}, index=index)
    for close, hold, expected in (
        ([100.0, 103.0, 100.0, 100.0], 10, [0, 1, 0, 0]),
        ([100.0, 100.0, 100.0, 100.0], 2, [0, 1, 1, 0]),
    ):
        data = pd.DataFrame({"open": 100.0, "close": close}, index=index)
        result = strategy.generate_signals(
            data, ready, ctx, StrategyParams(max_hold=hold)
        ).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
