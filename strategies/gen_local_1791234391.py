from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    roc_bars: int = 3
    atr_bars: int = 14
    compression_bars: int = 4
    baseline_bars: int = 20
    acceleration_threshold: float = 0.6
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.roc_bars, self.atr_bars, self.compression_bars, self.baseline_bars, self.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (np.isfinite(self.acceleration_threshold) and self.acceleration_threshold > 0):
            raise ValueError("Acceleration threshold must be finite and positive.")
        if not (np.isfinite(self.stop_atr) and self.stop_atr > 0):
            raise ValueError("Stop multiplier must be finite and positive.")
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1791234391"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(max(2 * params.roc_bars + 1, params.atr_bars + 1,
                       params.compression_bars + 1, params.baseline_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["atr_fraction"] = (out["atr"] / close).where(out["atr"] > 0)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["previous_roc"] = out["roc"].shift(params.roc_bars)
        out["acceleration"] = (out["roc"] - out["previous_roc"]) / (
            out["atr_fraction"] * np.sqrt(params.roc_bars))
        out["range_fraction"] = (data["high"] - data["low"]) / close
        out["recent_range"] = out["range_fraction"].shift(1).rolling(params.compression_bars).mean()
        out["baseline_range"] = out["range_fraction"].shift(1).rolling(params.baseline_bars).mean()
        out["compression"] = out["recent_range"] / out["baseline_range"].where(out["baseline_range"] > 0)
        out["previous_speed"] = out["previous_roc"].abs() / (
            out["atr_fraction"] * np.sqrt(params.roc_bars))
        out["strength"] = out["acceleration"].abs()
        ready = np.isfinite(out[["atr", "roc", "acceleration", "compression", "previous_speed"]]).all(axis=1)
        release = ready & (out["compression"] < 0.85) & (out["previous_speed"] < 0.5)
        bullish = release & (out["roc"] > 0) & (out["acceleration"] >= params.acceleration_threshold)
        bearish = release & (out["roc"] < 0) & (out["acceleration"] <= -params.acceleration_threshold)
        out["entry"] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].to_numpy(dtype=float)
        position, entry_bar = 0, -1
        fixed_stop, position_size = np.nan, params.min_size
        for i in range(count):
            if position:
                if i == entry_bar:
                    # The prior close's decision fills at this open; ATR is frozen.
                    if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                        position = 0
                        continue
                    fixed_stop = opens[i] - position * params.stop_atr * atr[i - 1]
                stopped = (not np.isfinite(closes[i])) or position * (closes[i] - fixed_stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
                    continue
            elif entries[i] and np.isfinite(strengths[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                conviction = float(np.clip(strengths[i] / (2 * params.acceleration_threshold), 0, 1))
                position_size = params.min_size + (params.max_size - params.min_size) * conviction
            signals[i] = position
            if position:
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        opens = 100 + side * np.array([0, 5, 4, 3, 2, 1], dtype=float)
        closes = 100 + side * np.array([0, 4, 10, 2, 1, 0], dtype=float)
        data = pd.DataFrame({"open": opens, "close": closes}, index=index)
        ind = pd.DataFrame({"atr": 1.0, "entry": [side, 0, 0, 0, 0, 0],
                            "strength": 0.6}, index=index)
        result = strategy.generate_signals(data, ind, None, ShockwaveParams()).data
        assert result["signal"].tolist() == [0, side, side, side, 0, 0]
        assert np.allclose(result.loc[result["signal"] != 0, "size"], 0.6)
        timed = strategy.generate_signals(data, ind, None, ShockwaveParams(hold_bars=2)).data
        assert timed["signal"].tolist() == [0, side, side, 0, 0, 0]
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=60))
    params = ShockwaveParams()
    ind = strategy.indicators(flat, params)
    assert ind["entry"].eq(0).all()
    assert np.isfinite(ind.iloc[strategy.warmup_bars(params) - 1]["acceleration"])


if __name__ == "__main__":
    _self_check()
