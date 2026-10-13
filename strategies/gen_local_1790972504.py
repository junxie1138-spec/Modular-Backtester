from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 9
    roc_bars: int = 3
    library_bars: int = 378
    horizon_bars: int = 8
    neighbors: int = 12
    min_edge: float = 0.003
    max_distance: float = 0.9
    atr_bars: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    initial_stop_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790972504"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.atr_bars + 1, params.library_bars + params.horizon_bars + params.shape_bars + params.roc_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(p.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["energy"] = out["acceleration"].abs().rolling(p.shape_bars, min_periods=p.shape_bars).sum()
        denominator = out["energy"].replace(0.0, np.nan)
        block = max(1, p.shape_bars // 3)
        # Distribution of acceleration mass encodes episode phase without matching individual samples.
        out["early_mass"] = out["acceleration"].abs().shift(p.shape_bars - block).rolling(block, min_periods=block).sum() / denominator
        out["late_mass"] = out["acceleration"].abs().rolling(block, min_periods=block).sum() / denominator
        out["balance"] = out["acceleration"].rolling(p.shape_bars, min_periods=p.shape_bars).sum() / denominator
        out["endpoint"] = (out["acceleration"] * p.shape_bars / denominator).clip(-3.0, 3.0) / 3.0
        out["roc_scale"] = out["roc"].abs().rolling(p.shape_bars, min_periods=p.shape_bars).max()
        out["roc_phase"] = out["roc"] / out["roc_scale"].replace(0.0, np.nan)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        # Outcomes are indexed by their maturity date, never by their starting date.
        origin = close.shift(p.horizon_bars)
        out["mature_return"] = close / origin - 1.0
        out["mature_upside"] = (data["high"].rolling(p.horizon_bars, min_periods=p.horizon_bars).max() / origin - 1.0).clip(lower=0.0)
        out["mature_downside"] = (1.0 - data["low"].rolling(p.horizon_bars, min_periods=p.horizon_bars).min() / origin).clip(lower=0.0)
        features = ["early_mass", "late_mass", "balance", "endpoint", "roc_phase"]
        distances, returns, upsides, downsides = [], [], [], []
        for lag in range(p.horizon_bars + 1, p.horizon_bars + p.library_bars + 1):
            delta = out[features] - out[features].shift(lag)
            distance = np.sqrt(delta.pow(2).sum(axis=1, min_count=len(features))).to_numpy()
            maturity_lag = lag - p.horizon_bars
            distances.append(distance)
            returns.append(out["mature_return"].shift(maturity_lag).to_numpy())
            upsides.append(out["mature_upside"].shift(maturity_lag).to_numpy())
            downsides.append(out["mature_downside"].shift(maturity_lag).to_numpy())
        dist = np.column_stack(distances)
        ret = np.column_stack(returns)
        up = np.column_stack(upsides)
        down = np.column_stack(downsides)
        valid = np.isfinite(dist) & np.isfinite(ret) & np.isfinite(up) & np.isfinite(down)
        dist = np.where(valid, dist, np.inf)
        k = min(max(1, p.neighbors), p.library_bars)
        selected = np.argpartition(dist, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(dist, selected, axis=1)
        r = np.take_along_axis(ret, selected, axis=1)
        u = np.take_along_axis(up, selected, axis=1)
        a = np.take_along_axis(down, selected, axis=1)
        accepted = np.isfinite(d) & (d <= p.max_distance)
        weights = np.where(accepted, np.exp(-d / max(p.max_distance, 1e-12)), 0.0)
        total = weights.sum(axis=1)
        divisor = np.where(total > 0.0, total, 1.0)
        out["analog_count"] = accepted.sum(axis=1)
        out["long_edge"] = (weights * np.where(accepted, r - 0.5 * a, 0.0)).sum(axis=1) / divisor
        out["short_edge"] = (weights * np.where(accepted, -r - 0.5 * u, 0.0)).sum(axis=1) / divisor
        out["long_agreement"] = (weights * (r > 0.0)).sum(axis=1) / divisor
        out["short_agreement"] = (weights * (r < 0.0)).sum(axis=1) / divisor
        ready = (out["analog_count"] >= k) & np.isfinite(out["atr"]) & (out["atr"] > 0.0)
        long_ok = ready & (out["long_edge"] > p.min_edge) & (out["long_agreement"] >= 0.6)
        short_ok = ready & (out["short_edge"] > p.min_edge) & (out["short_agreement"] >= 0.6)
        out["candidate"] = np.select([long_ok, short_ok], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        p = params
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        commands = np.zeros(len(data), dtype=int)
        side = 0
        pending = False
        armed = False
        entry = best = stop = entry_atr = 0.0
        entry_bar = 0
        for i in range(len(data)):
            exited = False
            if side != 0:
                if pending:
                    entry = prices[i, 0]
                    entry_bar = i
                    best = side * entry
                    stop = best - p.initial_stop_atr * entry_atr
                    pending = False
                o, high, low, last = prices[i]
                if not (np.isfinite(prices[i]).all() and np.isfinite(atr[i]) and atr[i] > 0.0):
                    side = 0
                    exited = True
                else:
                    # Test the previously established stop before using this bar's extremes.
                    adverse = min(side * low, side * high)
                    hit = adverse <= stop
                    best = max(best, side * high, side * low)
                    armed = armed or best - side * entry >= entry * p.breakeven_pct
                    if armed:
                        # Signed-price stop only rises; a short's actual stop only falls.
                        stop = max(stop, side * entry, best - p.trail_atr * atr[i])
                        hit = hit or side * last <= stop
                    if hit or i - entry_bar + 1 >= p.max_hold_bars:
                        side = 0
                        exited = True
            if side == 0 and not exited and i > 0:
                if candidate[i] != 0 and candidate[i] == candidate[i - 1]:
                    side = int(candidate[i])
                    pending = True
                    armed = False
                    entry_atr = float(atr[i])
            commands[i] = side
        df = pd.DataFrame({"signal": commands, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
