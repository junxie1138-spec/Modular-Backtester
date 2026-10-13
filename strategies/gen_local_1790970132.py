from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRatchetParams:
    channel_bars: int = 20
    ratchet_bars: int = 5
    volume_bars: int = 10
    atr_bars: int = 14
    min_floor_atr: float = 0.25
    max_selling_share: float = 0.35
    stop_atr: float = 2.5
    hold_bars: int = 18
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[RangeRatchetParams]):
    strategy_id = "gen_local_1790970132"

    @classmethod
    def params_type(cls):
        return RangeRatchetParams

    def warmup_bars(self, params):
        windows = (params.channel_bars, params.ratchet_bars, params.volume_bars, params.atr_bars)
        if min(windows) < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (params.min_floor_atr > 0 and 0 < params.max_selling_share < 1
                and params.stop_atr > 0 and params.entry_slippage_bps >= 0):
            raise ValueError("Invalid range, volume, stop, or slippage parameters.")
        return int(max(params.channel_bars + 1, params.ratchet_bars + 2,
                       params.volume_bars + 2, params.atr_bars + 2))

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["ceiling"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        out["floor_lift"] = data["low"].diff(params.ratchet_bars).shift(1)
        out["cap_lift"] = data["high"].diff(params.ratchet_bars).shift(1)
        out["previous_atr"] = out["atr"].shift(1)
        out["close_change"] = data["close"].diff()
        volume = data["volume"].where(data["volume"] >= 0)
        out["valid_volume"] = volume.where(out["close_change"].notna())
        out["selling_volume"] = out["valid_volume"].where(out["close_change"] < 0, 0.0).where(out["close_change"].notna())
        out["total_volume"] = out["valid_volume"].rolling(params.volume_bars, min_periods=params.volume_bars).sum().shift(1)
        out["total_selling_volume"] = out["selling_volume"].rolling(params.volume_bars, min_periods=params.volume_bars).sum().shift(1)
        out["selling_share"] = out["total_selling_volume"] / out["total_volume"].where(out["total_volume"] > 0)
        out["range_agrees"] = (
            (data["close"] > out["ceiling"])
            & (out["floor_lift"] > params.min_floor_atr * out["previous_atr"])
            & (out["floor_lift"] > out["cap_lift"])
            & (out["cap_lift"] >= 0)
        ).fillna(False)
        out["volume_agrees"] = (out["selling_share"] < params.max_selling_share).fillna(False)
        out["entry"] = (
            out["range_agrees"] & out["volume_agrees"]
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        self.warmup_bars(params)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        pending_atr = 0.0
        stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # A close decision fills at the following open; never read a future bar.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i] * (1.0 + params.entry_slippage_bps / 10000.0)
                    stop = entry_price - params.stop_atr * pending_atr
                    entry_bar = i
                    active = True
            if active:
                if (np.isfinite(close[i]) and close[i] < stop) or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                pending_atr = atr[i]
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeRatchetParams(stop_atr=2.0, hold_bars=18, entry_slippage_bps=0.0)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 131.0, "low": 94.0,
        "close": [100.0, 100.0, 130.0, 99.0, 95.0, 95.0], "volume": 100.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    indicators = pd.DataFrame({"atr": [2.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                               "entry": [True, False, False, False, False, False]}, index=index)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    params.hold_bars = 3
    data["close"] = 100.0
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
