from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 12
    atr_window: int = 14
    pressure_window: int = 3
    innovation_threshold: float = 0.035
    pressure_threshold: float = 0.18
    breakeven_pct: float = 0.008
    trail_atr: float = 1.4
    initial_stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791289107"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.slope_window + 2, params.atr_window + 1,
                       params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.slope_window < 3 or params.atr_window < 1
                or params.pressure_window < 1 or not 3 <= params.hold_bars <= 5
                or params.innovation_threshold <= 0
                or not 0 < params.pressure_threshold < 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0
                or params.initial_stop_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        length = params.slope_window
        out["log_sum"] = out["log_close"].rolling(
            length, min_periods=length).sum()
        weights = np.arange(length - 1, -1, -1, dtype=float)
        weighted = (np.convolve(out["log_close"].to_numpy(), weights,
                                mode="full")[:len(data)]
                    if len(data) else np.array([], dtype=float))
        out["weighted_log_sum"] = pd.Series(weighted, index=data.index).where(
            out["log_sum"].notna())
        out["slope"] = (out["weighted_log_sum"]
                        - (length - 1) * 0.5 * out["log_sum"]) / (
                            length * (length * length - 1) / 12.0)
        out["forecast_slope"] = 2.0 * out["slope"].shift(1) - out["slope"].shift(2)
        out["relative_atr"] = (out["atr"] / close).replace(0, np.nan)
        out["innovation"] = (
            out["slope"] - out["forecast_slope"]) / out["relative_atr"]
        volume = data["volume"].clip(lower=0)
        out["signed_turnover"] = (data["close"] - data["open"]) * volume
        out["range_turnover"] = (data["high"] - data["low"]).clip(lower=0) * volume
        out["pressure_numerator"] = out["signed_turnover"].rolling(
            params.pressure_window, min_periods=params.pressure_window).sum()
        out["pressure_denominator"] = out["range_turnover"].rolling(
            params.pressure_window, min_periods=params.pressure_window).sum()
        out["pressure"] = out["pressure_numerator"] / (
            out["pressure_denominator"].replace(0, np.nan))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        side = 0
        armed = True
        activated = False
        entry = stop = 0.0
        entry_bar = 0
        values = indicators[["slope", "innovation", "pressure", "atr"]].to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        for i, (slope, innovation, pressure, atr) in enumerate(values):
            valid = bool(np.isfinite([slope, innovation, pressure, atr,
                                      opens[i], closes[i]]).all()
                         and atr > 0 and opens[i] > 0 and closes[i] > 0)
            if np.isfinite(innovation) and abs(innovation) < params.innovation_threshold * 0.5:
                armed = True
            # A prior close's order enters at this bar's open.
            if side == 0 and i > 0 and raw[i - 1] != 0:
                if not valid:
                    continue
                side = int(raw[i - 1])
                entry = opens[i]
                entry_bar = i
                activated = False
                prior_atr = values[i - 1, 3]
                # Favorable-price coordinates make both directions ratchet upward.
                stop = side * entry - params.initial_stop_atr * prior_atr
            if side != 0:
                if not valid:
                    side = 0
                    continue
                favorable_close = side * closes[i]
                # Close-observed stops submit an exit for the next open.
                if favorable_close <= stop or i - entry_bar + 1 >= params.hold_bars:
                    side = 0
                    continue
                if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                    activated = True
                if activated:
                    stop = max(stop, side * entry,
                               favorable_close - params.trail_atr * atr)
                raw[i] = side
            elif valid and armed:
                long_entry = (slope > 0 and innovation > params.innovation_threshold
                              and pressure > params.pressure_threshold)
                short_entry = (slope < 0 and innovation < -params.innovation_threshold
                               and pressure < -params.pressure_threshold)
                if long_entry or short_entry:
                    raw[i] = 1 if long_entry else -1
                    armed = False
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=8)
    params = Params(hold_bars=5, trail_atr=0.25)
    for direction in (1, -1):
        close = np.array([100, 100, 101, 100.8, 100.6, 100, 100, 100])
        if direction == -1:
            close = 200 - close
        data = pd.DataFrame({"open": 100.0, "high": close + 2,
                             "low": close - 2, "close": close,
                             "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"slope": direction * 0.001,
                                   "innovation": direction * 0.1,
                                   "pressure": direction * 0.5,
                                   "atr": 1.0}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0, 0]
        data["close"] = 100.0
        result = strategy.generate_signals(
            data, indicators, None, Params(hold_bars=4)).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0, 0]
    data["close"] = np.exp(np.arange(8) * 0.01)
    actual = strategy.indicators(data, Params(slope_window=3, atr_window=2))
    assert np.allclose(actual["slope"].dropna(), 0.01)
    assert strategy.warmup_bars(Params(slope_window=3, atr_window=2)) == 5
    assert actual["innovation"].iloc[:4].isna().all()


if __name__ == "__main__":
    _self_check()
