from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FloorParams:
    floor_window: int = 5
    floor_lag: int = 10
    rank_window: int = 126
    rank_threshold: float = 0.80
    pressure_window: int = 10
    pressure_threshold: float = 0.62
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 3.0
    initial_stop_atr: float = 2.5
    max_hold: int = 18

    def __post_init__(self):
        if min(self.floor_window, self.floor_lag, self.rank_window, self.pressure_window, self.atr_window, self.max_hold) < 1:
            raise ValueError("Windows and holding period must be positive.")
        if not (0 < self.rank_threshold <= 1 and 0 < self.pressure_threshold <= 1):
            raise ValueError("Percentile and pressure thresholds must be in (0, 1].")
        if min(self.breakeven_pct, self.trail_atr, self.initial_stop_atr) <= 0:
            raise ValueError("Stop parameters must be positive.")


class GeneratedStrategy(BaseStrategy[FloorParams]):
    strategy_id = "gen_local_1791230236"

    @classmethod
    def params_type(cls):
        return FloorParams

    def warmup_bars(self, params: FloorParams) -> int:
        return int(max(params.floor_window + params.floor_lag + params.rank_window - 1,
                       params.atr_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: FloorParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["floor"] = data["low"].rolling(params.floor_window, min_periods=params.floor_window).min()
        out["floor_rise"] = out["floor"].diff(params.floor_lag) / out["atr"].replace(0, np.nan)
        out["floor_rank"] = out["floor_rise"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["close_location"] = ((data["close"] - data["low"]) /
                                 (data["high"] - data["low"]).replace(0, np.nan)).clip(0, 1).fillna(0.5)
        out["valid_volume"] = data["volume"].clip(lower=0).fillna(0.0)
        out["weighted_pressure"] = out["close_location"] * out["valid_volume"]
        out["pressure_sum"] = out["weighted_pressure"].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out["volume_sum"] = out["valid_volume"].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out["pressure"] = out["pressure_sum"] / out["volume_sum"].replace(0, np.nan)
        out["entry"] = (
            out["floor_rise"].gt(0) &
            out["floor_rank"].ge(params.rank_threshold) &
            out["pressure"].ge(params.pressure_threshold) &
            out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: FloorParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry_price = peak = stop = entry_atr = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    peak = entry_price
                    stop = entry_price - params.initial_stop_atr * entry_atr
                    entry_bar = i
                    armed = False
                    active = True
            if active:
                close = closes[i]
                # Close-observed stops are executed at the following open.
                if not np.isfinite(close) or close <= 0:
                    raw[i] = 1
                    continue
                breached = close <= stop
                peak = max(peak, close)
                if close >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atrs[i])
                if breached or close <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                entry_atr = atrs[i]
                pending = True
                raw[i] = 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0, 101.0, 104.0, 105.0, 100.5],
        "high": [101.0, 105.0, 106.0, 106.0, 107.0],
        "low": [99.0, 100.0, 103.0, 100.0, 100.0],
        "close": [100.0, 104.0, 105.0, 100.5, 106.0],
        "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 4.0, 4.0, 1.0],
        "entry": [True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, FloorParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    expired = strategy.generate_signals(
        data, indicators, None, FloorParams(max_hold=2, breakeven_pct=1.0)
    ).data
    assert expired["signal"].tolist() == [0, 1, 1, 0, 0]
    assert not strategy.indicators(data, FloorParams())["entry"].any()
    assert strategy.warmup_bars(FloorParams()) == 140


if __name__ == "__main__":
    _self_check()
