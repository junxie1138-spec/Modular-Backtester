from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 6
    atr_period: int = 14
    area_threshold: float = 0.12
    efficiency_threshold: float = 0.25
    breakeven_pct: float = 0.005
    trail_atr: float = 1.2
    max_hold: int = 2
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791259829"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.window + 1, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.window < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; window must be at least two.")
        if params.max_hold not in (1, 2):
            raise ValueError("max_hold must be one or two sessions.")
        values = np.array([params.area_threshold, params.efficiency_threshold,
                           params.breakeven_pct, params.trail_atr, params.minimum_size])
        if not np.isfinite(values).all():
            raise ValueError("Parameters must be finite.")
        if not (0 < params.area_threshold < 1 and 0 <= params.efficiency_threshold <= 1
                and params.breakeven_pct > 0 and params.trail_atr > 0
                and 0 < params.minimum_size <= 1):
            raise ValueError("Invalid thresholds, stop parameters, or sizing floor.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["up"] = out["return"].clip(lower=0)
        out["down"] = (-out["return"]).clip(lower=0)
        out["cum_up"] = out["up"].fillna(0).cumsum()
        out["cum_down"] = out["down"].fillna(0).cumsum()
        out["cross"] = (out["cum_up"].shift(1) * out["down"]
                        - out["cum_down"].shift(1) * out["up"])
        w = params.window
        out["sum_up"] = out["up"].rolling(w, min_periods=w).sum()
        out["sum_down"] = out["down"].rolling(w, min_periods=w).sum()
        out["variation"] = out["sum_up"] + out["sum_down"]
        out["area"] = (out["cross"].rolling(w, min_periods=w).sum()
                       - out["cum_up"].shift(w) * out["sum_down"]
                       + out["cum_down"].shift(w) * out["sum_up"])
        denominator = out["variation"].replace(0, np.nan)
        out["strength"] = (-4 * out["area"] / denominator.pow(2)).clip(-1, 1)
        out["net_move"] = out["sum_up"] - out["sum_down"]
        out["efficiency"] = out["net_move"].abs() / denominator
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.minimum_size, dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        efficiency = indicators["efficiency"].to_numpy(dtype=float)
        net_move = indicators["net_move"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        position = 0
        entry = 0.0
        stop_profit = 0.0
        best_profit = 0.0
        armed = False
        age = 0
        held_size = params.minimum_size

        for t in range(n):
            opening, high, low, close = prices[t]
            valid_price = (np.isfinite(prices[t]).all() and low > 0
                           and low <= min(opening, close)
                           and high >= max(opening, close))
            ready = (valid_price and np.isfinite([strength[t], efficiency[t], net_move[t], atr[t]]).all()
                     and atr[t] > 0)
            # A decision from the previous close enters at today's open.
            if position == 0 and t > 0 and signal[t - 1] != 0:
                if not valid_price or not np.isfinite(atr[t - 1]) or atr[t - 1] <= 0:
                    continue
                position = int(signal[t - 1])
                entry = opening
                held_size = size[t - 1]
                stop_profit = -params.trail_atr * atr[t - 1]
                best_profit = 0.0
                armed = False
                age = 0

            if position != 0:
                age += 1
                if not ready:
                    position = 0
                    continue
                # Express stops in directional profit units: they only increase.
                worst_profit = position * ((low if position == 1 else high) - entry)
                if worst_profit <= stop_profit or age >= params.max_hold:
                    position = 0
                    continue
                close_profit = position * (close - entry)
                best_profit = max(best_profit, close_profit)
                if close_profit >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop_profit = max(stop_profit, 0.0, best_profit - params.trail_atr * atr[t])
                # Close-armed stop updates take effect on subsequent bars.
                signal[t] = position
                size[t] = held_size
                continue

            if ready and abs(strength[t]) >= params.area_threshold:
                direction = 1 if strength[t] > 0 else -1
                if direction * net_move[t] > 0 and efficiency[t] >= params.efficiency_threshold:
                    signal[t] = direction
                    size[t] = params.minimum_size + (1.0 - params.minimum_size) * abs(strength[t])

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        assert df["signal"].isin([-1, 0, 1]).all()
        assert np.isfinite(df["size"]).all() and (df["size"] > 0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
