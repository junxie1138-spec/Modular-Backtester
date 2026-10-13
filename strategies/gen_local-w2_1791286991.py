from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    compression_bars: int = 3
    baseline_bars: int = 20
    atr_bars: int = 14
    compression_max: float = 1.8
    volume_min: float = 1.3
    stop_atr: float = 1.0
    hold_bars: int = 2
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791286991"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.compression_bars, params.baseline_bars, params.atr_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (min(params.compression_bars, params.baseline_bars, params.atr_bars) < 1
                or params.compression_max <= 0 or params.volume_min <= 0
                or params.stop_atr <= 0 or params.hold_bars not in (1, 2)
                or params.entry_slippage_bps < 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["daily_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["range_baseline"] = out["true_range"].rolling(params.baseline_bars).mean().shift(1)
        out["upper"] = data["high"].rolling(params.compression_bars).max().shift(1)
        out["lower"] = data["low"].rolling(params.compression_bars).min().shift(1)
        out["channel_range"] = out["upper"] - out["lower"]
        out["compression"] = out["channel_range"] / out["range_baseline"].replace(0, np.nan)
        out["volume_baseline"] = data["volume"].rolling(params.baseline_bars).mean().shift(1)
        out["loaded_volume"] = data["volume"].rolling(params.compression_bars).mean().shift(1)
        out["loading_ratio"] = out["loaded_volume"] / out["volume_baseline"].replace(0, np.nan)
        out["volume_ratio"] = data["volume"] / out["volume_baseline"].replace(0, np.nan)
        out["close_location"] = (data["close"] - data["low"]) / out["daily_range"].replace(0, np.nan)
        out["penetration"] = (data["close"] - out["upper"]) / out["prior_atr"].replace(0, np.nan)
        out["strength"] = (
            (out["volume_ratio"] / params.volume_min - 1).clip(0, 1)
            + out["penetration"].clip(0, 1)
        ) / 2
        ready = np.isfinite(out).all(axis=1)
        out["entry"] = (
            ready
            & (out["atr"] > 0)
            & (out["compression"] <= params.compression_max)
            & (out["loading_ratio"] >= 1.0)
            & (out["volume_ratio"] >= params.volume_min)
            & (out["close_location"] >= 0.75)
            & (data["close"] > out["upper"])
            & (data["close"] > data["open"])
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].fillna(0).to_numpy(dtype=float)
        active = False
        entry_bar = -1
        stop = np.nan
        frozen_atr = 0.0
        position_size = 1.0
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i] * (1 + params.entry_slippage_bps / 10000)
                    stop = entry_price - params.stop_atr * frozen_atr
                if (not np.isfinite(closes[i]) or not np.isfinite(stop)
                        or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
            elif (entries[i] and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(strength[i])):
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                position_size = float(np.clip(0.25 + 0.75 * strength[i], 0.25, 1.0))
            signals[i] = int(active)
            sizes[i] = position_size if active else 1.0
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(entry_slippage_bps=0.0)
    index = pd.date_range("2020-01-01", periods=4)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    ind = pd.DataFrame({
        "entry": [True, False, False, False],
        "atr": [1.0, 100.0, 100.0, 100.0],
        "strength": [0.75, 0.0, 0.0, 0.0],
    }, index=index)
    for closes, expected in (
        ([100.0, 108.0, 110.0, 110.0], [0, 1, 0, 0]),
        ([100.0, 111.0, 111.0, 111.0], [0, 1, 1, 0]),
    ):
        data = pd.DataFrame({
            "open": [100.0, 110.0, 110.0, 110.0], "close": closes,
            "high": [112.0] * 4, "low": [99.0] * 4, "volume": [1000.0] * 4,
        }, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == expected
        assert result["size"].iloc[1] == 0.8125
        for n in range(1, 5):
            prefix = strategy.generate_signals(data.iloc[:n], ind.iloc[:n], ctx, params).data
            pd.testing.assert_frame_equal(prefix, result.iloc[:n])
    empty_ind = strategy.indicators(data, params)
    assert not empty_ind["entry"].any()
    assert np.isfinite(empty_ind.to_numpy(dtype=float)).all()


if __name__ == "__main__":
    _self_check()
