from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_window: int = 40
    normalization_window: int = 40
    drawdown_window: int = 30
    min_drawdown: float = 0.025
    excursion_z: float = 1.2
    retest_band: float = 0.005
    memory_recovery: float = 0.5
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790982896"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.drawdown_window,
                       params.frac_window + params.normalization_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0 < params.frac_order < 1 and params.frac_window >= 2
                and params.normalization_window >= 2 and params.drawdown_window >= 2
                and 0 < params.min_drawdown < 1 and params.excursion_z > 0
                and 0 < params.retest_band < 1 and params.memory_recovery > 0
                and params.profit_target > 0 and params.max_hold >= 1):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float).where(data["close"] > 0)
        weights = np.empty(params.frac_window)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        fd = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fd = fd + weight * close.shift(k)
        mean = fd.rolling(params.normalization_window).mean()
        std = fd.rolling(params.normalization_window).std().replace(0.0, np.nan)
        peak = close.rolling(params.drawdown_window).max()
        return pd.DataFrame({
            "frac_close": fd,
            "frac_change": fd.diff(),
            "frac_std": std,
            "frac_z": (fd - mean) / std,
            "drawdown": close / peak - 1.0,
            "ma200": close.rolling(200).mean(),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        values = indicators[["frac_close", "frac_change", "frac_std",
                             "frac_z", "drawdown", "ma200"]].to_numpy(dtype=float)
        held = False
        armed_at = -1
        entry_bar = -1
        entry_price = np.nan
        low_price = low_memory = np.nan
        for i in range(len(data)):
            price = close[i]
            if held:
                if i == entry_bar:
                    entry_price = opening[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        held = False
                        continue
                elapsed = i - entry_bar + 1
                target_hit = np.isfinite(price) and price >= entry_price * (1 + params.profit_target)
                if target_hit or elapsed >= params.max_hold:
                    held = False
                    continue
                signal[i] = 1
                continue
            fd, change, std, z, drawdown, ma = values[i]
            if not np.isfinite(price) or price <= 0 or not np.isfinite(values[i]).all():
                armed_at = -1
                continue
            if price <= ma:
                armed_at = -1
                continue
            if armed_at >= 0 and i - armed_at > 2 * params.max_hold:
                armed_at = -1
            if armed_at < 0:
                if drawdown <= -params.min_drawdown and z <= -params.excursion_z:
                    armed_at, low_price, low_memory = i, price, fd
                continue
            retest = abs(price / low_price - 1.0) <= params.retest_band
            repaired = fd >= low_memory + params.memory_recovery * std
            if retest and repaired and change > 0 and drawdown <= -params.min_drawdown:
                held = True
                entry_bar = i + 1
                entry_price = np.nan
                armed_at = -1
                signal[i] = 1
            elif price < low_price:
                low_price, low_memory = price, fd
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 97, 97, 98, 100, 100, 100],
        "close": [100, 97, 97.2, 100, 100, 100, 100],
    }, index=index)
    ind = pd.DataFrame({
        "frac_close": [-3, -3, -2, 0, 0, 0, 0],
        "frac_change": [0, 0, 1, 2, 0, 0, 0],
        "frac_std": 1.0, "frac_z": -3.0,
        "drawdown": -0.04, "ma200": 90.0,
    }, index=index)
    for target, expected in [
        (0.02, [0, 0, 0, 1, 0, 0, 0]),
        (0.50, [0, 0, 0, 1, 1, 1, 0]),
    ]:
        result = GeneratedStrategy.generate_signals(
            None, data, ind, None, Params(profit_target=target, max_hold=3))
        assert result.data["signal"].tolist() == expected
        assert result.data["size"].gt(0).all()
    blocked = ind.copy()
    blocked["ma200"] = 110.0
    result = GeneratedStrategy.generate_signals(None, data, blocked, None, Params())
    assert result.data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
