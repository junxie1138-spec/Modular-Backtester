from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    position_window: int = 20
    area_window: int = 12
    atr_period: int = 14
    area_threshold: float = 0.08
    entry_band: float = 0.35
    atr_multiple: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790965045"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.position_window + params.area_window + 1,
                       params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if (min(params.position_window, params.area_window, params.atr_period) < 2
                or params.max_hold < 1 or not 0.0 < params.entry_band < 0.5
                or not np.isfinite(params.area_threshold) or params.area_threshold <= 0
                or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["range_high"] = data["high"].rolling(params.position_window).max()
        out["range_low"] = data["low"].rolling(params.position_window).min()
        width = (out["range_high"] - out["range_low"]).replace(0.0, np.nan)
        out["position"] = ((close - out["range_low"]) / width).clip(0.0, 1.0)
        daily_width = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["pressure"] = ((close - data["low"]) / daily_width).clip(0.0, 1.0)
        x, y = out["position"], out["pressure"]
        out["dx"], out["dy"] = x.diff(), y.diff()
        out["cross"] = x.shift(1) * out["dy"] - y.shift(1) * out["dx"]
        length = params.area_window
        out["sum_dx"] = out["dx"].rolling(length).sum()
        out["sum_dy"] = out["dy"].rolling(length).sum()
        out["sum_cross"] = out["cross"].rolling(length).sum()
        # Antisymmetric second-order signature, anchored at the window's first point.
        out["area"] = 0.5 * (out["sum_cross"] - x.shift(length) * out["sum_dy"]
                             + y.shift(length) * out["sum_dx"])
        out["ma200"] = close.rolling(200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        state, held = 0, 0
        watermark = stop = -np.inf
        values = indicators[["ma200", "atr", "position", "dx", "pressure", "area"]].to_numpy()
        for i, (close, row) in enumerate(zip(data["close"].to_numpy(), values)):
            ma, atr, location, dx, pressure, area = row
            if not np.isfinite([close, ma, atr]).all() or close <= 0 or atr <= 0:
                state, held = 0, 0
                watermark = stop = -np.inf
                continue
            if state:
                held += 1
                # Begin tracking on the first fill bar, not the prior decision bar.
                # Long: highest close and upward stop; short: lowest close and downward stop.
                favorable_close = state * close
                watermark = max(watermark, favorable_close)
                stop = max(stop, watermark - params.atr_multiple * atr)
                if (favorable_close <= stop or held >= params.max_hold
                        or state * (close - ma) <= 0):
                    state, held = 0, 0
                    watermark = stop = -np.inf
            elif np.isfinite(row).all():
                if (close > ma and location <= params.entry_band and dx > 0
                        and pressure >= 0.6 and area <= -params.area_threshold):
                    state = 1
                elif (close < ma and location >= 1.0 - params.entry_band and dx < 0
                      and pressure <= 0.4 and area >= params.area_threshold):
                    state = -1
                if state:
                    held = 0
                    watermark = stop = -np.inf
            raw[i] = state
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(atr_multiple=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [100, 101, 110, 109, 107, 106],
                         "high": [101, 102, 111, 110, 108, 107],
                         "low": [99, 100, 109, 108, 106, 105],
                         "close": [100, 101, 110, 109, 107, 106],
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"ma200": 90.0, "atr": [1, 1, 1, 10, 10, 10],
                        "position": 0.2, "dx": 0.1, "pressure": 0.8,
                        "area": [-0.2, 0, 0, 0, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    mirrored = data.copy()
    mirrored["close"] = 200.0 - data["close"]
    short_ind = ind.copy()
    short_ind["ma200"] = 110.0
    short_ind["position"], short_ind["dx"], short_ind["pressure"] = 0.8, -0.1, 0.2
    short_ind["area"] = -ind["area"]
    assert strategy.generate_signals(mirrored, short_ind, ctx, params).data["signal"].tolist() == [0, -1, -1, -1, -1, 0]
    timed = strategy.generate_signals(data, ind, ctx, GeneratedParams(max_hold=2, atr_multiple=100)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert strategy.indicators(data, params)["ma200"].isna().all()
    assert strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data["signal"].eq(0).all()
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
