from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeParams:
    range_lookback: int = 12
    atr_period: int = 14
    contraction_max: float = 0.70
    imbalance_min: float = 0.20
    edge_min: float = 0.60
    stop_atr: float = 0.90
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeParams]):
    strategy_id = "gen_local-w2_1790988994"

    @classmethod
    def params_type(cls) -> type[RangeParams]:
        return RangeParams

    def warmup_bars(self, params: RangeParams) -> int:
        return int(max(max(1, params.range_lookback), max(1, params.atr_period)) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["span"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["previous_span"] = out["span"].shift(1).replace(0.0, np.nan)
        out["range_reference"] = out["span"].rolling(max(1, params.range_lookback), min_periods=max(1, params.range_lookback)).mean().shift(1).replace(0.0, np.nan)
        out["upper_retreat"] = data["high"].shift(1) - data["high"]
        out["lower_retreat"] = data["low"] - data["low"].shift(1)
        out["imbalance"] = (out["upper_retreat"] - out["lower_retreat"]) / out["previous_span"]
        out["close_position"] = (2.0 * (data["close"] - data["low"]) / out["span"].replace(0.0, np.nan) - 1.0).clip(-1.0, 1.0)
        out["contraction"] = out["span"] / out["previous_span"]
        out["range_ratio"] = out["span"] / out["range_reference"]
        out["true_range"] = pd.concat([
            out["span"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(max(1, params.atr_period), min_periods=max(1, params.atr_period)).mean()
        out["strength"] = (out["imbalance"].abs() * out["close_position"].abs()).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeParams) -> SignalFrame:
        ind = indicators
        eligible = (
            (ind["upper_retreat"] >= 0.0)
            & (ind["lower_retreat"] >= 0.0)
            & (ind["span"] > 0.0)
            & (ind["contraction"] <= params.contraction_max)
            & (ind["range_ratio"] <= 1.0)
            & (ind["imbalance"].abs() >= params.imbalance_min)
            & (ind["close_position"].abs() >= params.edge_min)
            & (ind["imbalance"] * ind["close_position"] < 0.0)
            & (ind["atr"] > 0.0)
        ).fillna(False)
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & (ind["imbalance"] > 0.0)] = 1
        candidates.loc[eligible & (ind["imbalance"] < 0.0)] = -1
        conviction_size = (0.25 + 0.75 * (ind["strength"].fillna(0.0) / 0.60).clip(0.0, 1.0)).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = ind["atr"].to_numpy(dtype=float)
        proposed = candidates.to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side = 0
        age = 0
        fixed_stop = np.nan
        active_size = 1.0
        horizon = min(2, max(1, int(params.hold_bars)))
        stop_multiple = max(0.01, float(params.stop_atr))

        for i in range(len(data)):
            if side != 0:
                # The preceding close's order fills at this bar's open.
                if age == 0:
                    entry_atr = atr[i - 1] if i > 0 else np.nan
                    if not (np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(entry_atr) and entry_atr > 0.0):
                        side = 0
                        continue
                    fixed_stop = opens[i] - side * stop_multiple * entry_atr
                age += 1
                stopped = not np.isfinite(closes[i]) or (side == 1 and closes[i] < fixed_stop) or (side == -1 and closes[i] > fixed_stop)
                if stopped or age >= horizon:
                    side = 0
                    age = 0
                signals[i] = side
                sizes[i] = active_size
                continue

            if proposed[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                side = int(proposed[i])
                age = 0
                active_size = float(conviction_size[i])
                signals[i] = side
                sizes[i] = active_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
