from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_window: int = 14
    gap_window: int = 20
    volume_window: int = 20
    shock_z: float = 1.5
    pressure_min: float = 0.6
    stop_atr: float = 0.8
    holding_bars: int = 2

    def __post_init__(self):
        if min(self.atr_window, self.gap_window, self.volume_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if self.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be one or two.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.shock_z, self.pressure_min, self.stop_atr)):
            raise ValueError("Thresholds must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791293966"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_window, params.gap_window, params.volume_window) + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"].div(out["previous_close"].where(out["previous_close"] > 0)) - 1
        out["gap_std"] = out["gap"].rolling(params.gap_window, min_periods=params.gap_window).std(ddof=0).shift(1)
        out["gap_z"] = out["gap"].div(out["gap_std"].where(out["gap_std"] > 0))
        out["range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["volume_mean"] = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out["close_location"] = (2 * data["close"] - data["high"] - data["low"]).div(out["range"].where(out["range"] > 0))
        out["pressure"] = out["close_location"] * data["volume"].div(out["volume_mean"].where(out["volume_mean"] > 0))
        valid = np.isfinite(out[["gap_z", "pressure", "atr"]]).all(axis=1) & out["atr"].gt(0)
        buy = valid & out["gap_z"].le(-params.shock_z) & out["pressure"].ge(params.pressure_min)
        sell = valid & out["gap_z"].ge(params.shock_z) & out["pressure"].le(-params.pressure_min)
        out["entry"] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position, entry_bar = 0, -1
        entry_price, frozen_atr = np.nan, np.nan
        for i in range(len(data)):
            if position:
                # A decision on the prior close enters at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                stopped = valid and position * (closes[i] - entry_price) < -params.stop_atr * frozen_atr
                expired = i - entry_bar + 1 >= params.holding_bars
                if not valid or stopped or expired:
                    position = 0
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                frozen_atr = atrs[i]
                entry_price = np.nan
            desired[i] = position
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100., 110., 110., 100., 100., 90., 90., 100.],
        "close": [100., 108., 110., 100., 100., 92., 90., 100.],
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [1, 0, 0, 0, -1, 0, 0, 0],
        "atr": [1., 100., 100., 100., 1., 100., 100., 100.],
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(stop_atr=1.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, -1, 0, 0]
    data["close"] = data["open"]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= params.gap_window + 2


if __name__ == "__main__":
    _self_check()
