from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_order: float = 0.4
    fd_window: int = 32
    baseline_window: int = 20
    compression_window: int = 5
    compression_ratio: float = 0.8
    trigger_z: float = 1.2
    profit_target: float = 0.018
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791285151"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.fd_window + params.baseline_window + 2,
                   params.fd_window + params.compression_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        fd = log_close.copy()
        weight = 1.0
        # ponytail: fixed-width fractional memory; increase fd_window for longer memory.
        for lag in range(1, params.fd_window):
            weight *= (lag - 1 - params.fd_order) / lag
            fd = fd + weight * log_close.shift(lag)
        baseline = fd.rolling(params.baseline_window, min_periods=params.baseline_window)
        mean = baseline.mean().shift(1)
        scale = baseline.std(ddof=0).shift(1).replace(0.0, np.nan)
        short_scale = fd.rolling(params.compression_window,
                                 min_periods=params.compression_window).std(ddof=0).shift(1)
        z = ((fd - mean) / scale).replace([np.inf, -np.inf], np.nan)
        ma200 = close.rolling(200, min_periods=200).mean()
        compressed = short_scale < params.compression_ratio * scale
        entry = (compressed & (z > params.trigger_z)
                 & (z.shift(1) <= params.trigger_z) & (close > ma200))
        return pd.DataFrame({
            "log_close": log_close, "fractional_close": fd,
            "baseline": mean, "baseline_scale": scale,
            "short_scale": short_scale, "fractional_z": z,
            "ma200": ma200, "compressed": compressed.fillna(False),
            "entry": entry.fillna(False)
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if not 0.0 < params.profit_target or not 3 <= params.hold_bars <= 5:
            raise ValueError("profit_target must be positive; hold_bars must be 3 to 5")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
                else:
                    continue
            if active:
                # Gross target uses the entry open; close-confirmed exits fill next open.
                target_hit = (np.isfinite(closes[i])
                              and closes[i] >= entry_price * (1.0 + params.profit_target))
                time_up = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_up:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params(profit_target=0.02, hold_bars=4)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 102.1
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.loc[index[[0, 6]], "entry"] = True
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
