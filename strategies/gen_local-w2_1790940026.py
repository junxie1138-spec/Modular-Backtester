from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShockParams:
    roc_window: int = 5
    gap_window: int = 5
    acceleration_lag: int = 3
    atr_window: int = 14
    pressure_threshold: float = 0.6
    acceleration_threshold: float = 0.04
    breakeven_profit: float = 0.025
    initial_stop_atr: float = 2.5
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GapShockParams]):
    strategy_id = "gen_local-w2_1790940026"

    @classmethod
    def params_type(cls):
        return GapShockParams

    def warmup_bars(self, params: GapShockParams) -> int:
        return int(max(params.atr_window + 1,
                       params.roc_window + params.acceleration_lag + 1,
                       params.gap_window + params.acceleration_lag + 1))

    def indicators(self, data: pd.DataFrame, params: GapShockParams) -> pd.DataFrame:
        windows = (params.roc_window, params.gap_window,
                   params.acceleration_lag, params.atr_window, params.max_hold)
        if any(v < 1 for v in windows):
            raise ValueError("Lookbacks and max_hold must be positive.")
        thresholds = (params.pressure_threshold, params.acceleration_threshold,
                      params.breakeven_profit, params.initial_stop_atr, params.trail_atr)
        if any(not np.isfinite(v) or v <= 0 for v in thresholds):
            raise ValueError("Thresholds and stop distances must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["gap"] = data["open"].div(previous_close.where(previous_close > 0)).sub(1)
        out["roc"] = data["close"].pct_change(params.roc_window, fill_method=None)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["atr_fraction"] = out["atr"].div(data["close"].where(data["close"] > 0))
        scale = out["atr_fraction"].where(out["atr_fraction"] > 0)
        out["gap_flow"] = out["gap"].rolling(params.gap_window).mean()
        out["gap_pressure"] = out["gap"].rolling(params.gap_window).sum().div(scale)
        out["gap_acceleration"] = (
            out["gap_flow"].diff(params.acceleration_lag)
            .div(params.acceleration_lag).div(scale)
        )
        out["close_acceleration"] = (
            out["roc"].diff(params.acceleration_lag)
            .div(params.acceleration_lag).div(scale)
        )
        valid = np.isfinite(out[["gap_pressure", "gap_acceleration",
                                 "close_acceleration", "atr"]]).all(axis=1)
        long_setup = (
            valid & (out["gap_pressure"] < -params.pressure_threshold)
            & (out["gap_acceleration"] < -params.acceleration_threshold)
            & (out["close_acceleration"] > params.acceleration_threshold)
        )
        short_setup = (
            valid & (out["gap_pressure"] > params.pressure_threshold)
            & (out["gap_acceleration"] > params.acceleration_threshold)
            & (out["close_acceleration"] < -params.acceleration_threshold)
        )
        out["setup"] = long_setup.astype(int) - short_setup.astype(int)
        out["confirmed"] = out["setup"].where(
            out["setup"].eq(out["setup"].shift(1)), 0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapShockParams) -> SignalFrame:
        count = len(data)
        desired = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry = peak = stop = 0.0
        armed = False
        age = 0
        for i in range(count):
            valid = (
                np.isfinite([opens[i], highs[i], lows[i], closes[i], atr[i]]).all()
                and min(opens[i], highs[i], lows[i], closes[i], atr[i]) > 0
            )
            if not valid:
                position = 0
                continue
            # Yesterday's decision enters at today's open, matching the final shift.
            if position == 0 and i > 0 and desired[i - 1] != 0:
                position = int(desired[i - 1])
                entry = opens[i]
                peak = position * entry
                stop = peak - params.initial_stop_atr * atr[i - 1]
                armed = False
                age = 0
            if position != 0:
                age += 1
                # Signed prices let both long and short stops ratchet only upward.
                adverse = lows[i] if position == 1 else -highs[i]
                if adverse <= stop or age >= params.max_hold:
                    position = 0
                    continue
                favorable = highs[i] if position == 1 else -lows[i]
                peak = max(peak, favorable)
                if peak - position * entry >= params.breakeven_profit * entry:
                    armed = True
                if armed:
                    stop = max(stop, position * entry, peak - params.trail_atr * atr[i])
                    if position * closes[i] <= stop:
                        position = 0
                        continue
                desired[i] = position
            elif confirmed[i] != 0:
                desired[i] = int(confirmed[i])
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        # Stop touches are close-time exit decisions, filled at the next open.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapShockParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0, 0, 3, 2, 1, 0, 0, 0], dtype=float)
        data = pd.DataFrame(
            {"open": 100.0, "high": np.maximum(100.0, close) + 0.1,
             "low": np.minimum(100.0, close) - 0.1,
             "close": close, "volume": 1000.0},
            index=pd.date_range("2020-01-01", periods=len(close))
        )
        ind = pd.DataFrame({"atr": 1.0, "confirmed": 0}, index=data.index)
        ind.iloc[1, ind.columns.get_loc("confirmed")] = direction
        signals = strategy.generate_signals(data, ind, ctx, params).data
        assert signals["signal"].tolist() == [0, 0, direction, direction, 0, 0, 0, 0]
        assert signals["size"].gt(0).all()
        natural = strategy.indicators(data, params)
        assert natural["confirmed"].eq(0).all()
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9,
         "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=25)
    )
    ind = pd.DataFrame({"atr": 1.0, "confirmed": 0}, index=data.index)
    ind.iloc[1, ind.columns.get_loc("confirmed")] = 1
    signal = strategy.generate_signals(data, ind, ctx, params).data["signal"]
    assert signal.iloc[2:20].eq(1).all() and signal.iloc[20:].eq(0).all()
    natural = strategy.indicators(data, params)
    confirmed = natural["confirmed"].ne(0)
    assert (natural.loc[confirmed, "setup"]
            .eq(natural["setup"].shift(1).loc[confirmed])).all()
    assert strategy.warmup_bars(params) >= params.gap_window + params.acceleration_lag + 1


if __name__ == "__main__":
    _self_check()
