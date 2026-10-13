from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    atr_window: int = 14
    entry_z: float = 1.25
    spike_z: float = 2.25
    cooldown_bars: int = 6
    hold_bars: int = 4
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791278677"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(200, params.ma_window + params.cooldown_bars, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.ma_window < 2 or params.atr_window < 1
                or params.cooldown_bars < 1 or not 3 <= params.hold_bars <= 5
                or not np.isfinite([params.entry_z, params.spike_z, params.stop_atr]).all()
                or min(params.entry_z, params.spike_z, params.stop_atr) <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0).replace(0.0, np.nan)
        out["z"] = (close - out["ma"]) / out["std"]
        out["ma200"] = close.rolling(200).mean()
        previous_close = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["tr"].rolling(params.atr_window).mean()
        out["recent_peak_z"] = out["z"].rolling(params.cooldown_bars).max()
        out["entry"] = (
            (out["z"].shift(1) < -params.entry_z)
            & (out["z"] >= -params.entry_z)
            & (out["z"] < 0)
            & (close > out["ma200"])
            & (out["recent_peak_z"] < params.spike_z)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        stop = np.nan
        frozen_atr = np.nan
        for i in range(len(data)):
            if active:
                # The preceding close requested entry; this bar supplies its open.
                if i == entry_bar:
                    stop = opens[i] - params.stop_atr * frozen_atr
                if (not np.isfinite(closes[i]) or not np.isfinite(stop)
                        or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                    continue
                signals[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                active = True
                entry_bar = i + 1
                frozen_atr = atrs[i]
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 111.0, "low": 95.0,
        "close": [100.0, 100.0, 110.0, 99.0, 96.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [False, True, False, False, False, False, False, False],
        "atr": 2.0,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, StrategyParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    result = strategy.generate_signals(data, indicators, ctx, StrategyParams(stop_atr=100.0))
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert not strategy.indicators(flat, StrategyParams())["entry"].any()


if __name__ == "__main__":
    _self_check()
