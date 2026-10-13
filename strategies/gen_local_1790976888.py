from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSlopeParams:
    level_span: int = 8
    slope_span: int = 5
    atr_window: int = 14
    gap_min_atr: float = 0.15
    innovation_min: float = 0.30
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    max_hold: int = 4
    size_min: float = 0.25
    size_max: float = 0.95


class GeneratedStrategy(BaseStrategy[GapSlopeParams]):
    strategy_id = "gen_local_1790976888"

    @classmethod
    def params_type(cls):
        return GapSlopeParams

    def warmup_bars(self, params):
        return int(max(params.level_span + params.slope_span + 2, params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.level_span, params.slope_span, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["log_close"] = np.log(close)
        out["level"] = out["log_close"].ewm(
            span=params.level_span, adjust=False, min_periods=params.level_span
        ).mean()
        out["level_step"] = out["level"].diff()
        out["slope"] = out["level_step"].ewm(
            span=params.slope_span, adjust=False, min_periods=params.slope_span
        ).mean()
        out["prior_slope"] = out["slope"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["atr_fraction"] = out["atr"] / previous
        scale = out["atr_fraction"].where(out["atr_fraction"] > 0)
        out["slope_innovation"] = (out["slope"] - out["prior_slope"]) / scale
        out["gap_atr"] = (data["open"] - previous) / out["atr"].where(out["atr"] > 0)
        out["body"] = close - data["open"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        if not (0 < params.size_min <= params.size_max <= 1):
            raise ValueError("Position sizes must satisfy 0 < min <= max <= 1.")
        if min(params.innovation_min, params.gap_min_atr, params.breakeven_pct, params.trail_atr) <= 0 or params.max_hold < 1:
            raise ValueError("Thresholds and holding period must be positive.")
        gap = indicators["gap_atr"]
        innovation = indicators["slope_innovation"]
        ready = indicators[["atr", "prior_slope", "gap_atr", "slope_innovation", "body"]].notna().all(axis=1)
        eligible = (
            ready
            & (gap.abs() >= params.gap_min_atr)
            & (innovation.abs() >= params.innovation_min)
            & (gap * indicators["prior_slope"] < 0)
            & (gap * innovation > 0)
            & (gap * indicators["body"] > 0)
        )
        candidate = np.sign(gap.where(eligible, 0).fillna(0)).to_numpy(dtype=int)
        strength = (innovation.abs() / params.innovation_min - 1).clip(0, 2).fillna(0) / 2
        requested_size = (params.size_min + (params.size_max - params.size_min) * strength).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.size_min, dtype=float)
        direction, entry, held = 0, np.nan, 0
        stop, best, armed, position_size = -np.inf, -np.inf, False, params.size_min
        for i in range(len(data)):
            exited = False
            if direction:
                if np.isnan(entry):
                    # The preceding close decision fills at this bar's open.
                    entry = opens[i]
                    best = direction * entry
                held += 1
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(closes[i]):
                    direction, exited = 0, True
                else:
                    signed_close = direction * closes[i]
                    # Check the existing stop before incorporating today's close.
                    if signed_close <= stop or held >= params.max_hold:
                        direction, exited = 0, True
                    else:
                        best = max(best, signed_close)
                        if best - direction * entry >= entry * params.breakeven_pct:
                            armed = True
                        if armed:
                            # Signed-price stops rise for both directions;
                            # a short's corresponding dollar stop falls.
                            stop = max(stop, direction * entry)
                            if np.isfinite(atr[i]) and atr[i] > 0:
                                stop = max(stop, best - params.trail_atr * atr[i])
                            if signed_close <= stop:
                                direction, exited = 0, True
            if not direction and not exited and candidate[i]:
                direction = int(candidate[i])
                entry, held = np.nan, 0
                stop, best, armed = -np.inf, -np.inf, False
                position_size = float(requested_size[i])
            signals[i] = direction
            sizes[i] = position_size if direction else params.size_min
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.size_min).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy, params = GeneratedStrategy(), GapSlopeParams()
    index = pd.date_range("2020-01-01", periods=60)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].eq(0).all()
    for side in (1, -1):
        forced = pd.DataFrame({"atr": 1.0, "prior_slope": -side * 0.001, "gap_atr": side * 0.5, "slope_innovation": side * 0.9, "body": float(side)}, index=index)
        result = strategy.generate_signals(data, forced, ctx, params).data
        assert result["signal"].iloc[:6].tolist() == [0, side, side, side, side, 0]
        assert result["size"].between(params.size_min, params.size_max).all()
        moved = data.copy()
        moved.iloc[1, moved.columns.get_loc("close")] = 100 + side * 2
        stopped = strategy.generate_signals(moved, forced, ctx, params).data
        assert stopped["signal"].iloc[:4].tolist() == [0, side, side, 0]
    prefix = strategy.generate_signals(data.iloc[:40], ind.iloc[:40], ctx, params).data
    full = strategy.generate_signals(data, ind, ctx, params).data
    pd.testing.assert_frame_equal(prefix, full.iloc[:40])
