from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuantileMigrationParams:
    activation_pct: float = 0.025
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[QuantileMigrationParams]):
    strategy_id = "gen_local-w2_1790971033"

    @classmethod
    def params_type(cls) -> type[QuantileMigrationParams]:
        return QuantileMigrationParams

    def warmup_bars(self, params: QuantileMigrationParams) -> int:
        return 41

    def indicators(self, data: pd.DataFrame, params: QuantileMigrationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        out["returns"] = close.div(previous.where(previous > 0)).sub(1).replace([np.inf, -np.inf], np.nan)
        returns = out["returns"]
        out["recent_lower"] = returns.rolling(10, min_periods=10).quantile(0.25)
        out["recent_upper"] = returns.rolling(10, min_periods=10).quantile(0.75)
        out["baseline_lower"] = returns.rolling(30, min_periods=30).quantile(0.25).shift(10)
        out["baseline_upper"] = returns.rolling(30, min_periods=30).quantile(0.75).shift(10)
        out["baseline_width"] = out["baseline_upper"] - out["baseline_lower"]
        out["lower_migration"] = out["recent_lower"] - out["baseline_lower"]
        out["upper_migration"] = out["recent_upper"] - out["baseline_upper"]
        out["recent_return"] = close.div(close.shift(10).where(close.shift(10) > 0)).sub(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        margin = 0.15 * out["baseline_width"]
        upward = (out["lower_migration"] > margin) & (out["upper_migration"] > margin)
        downward = (out["lower_migration"] < -margin) & (out["upper_migration"] < -margin)
        widening = (out["lower_migration"] < -margin) & (out["upper_migration"] > margin)
        out["regime"] = np.select([upward, downward, widening], [1, -1, 2], default=0)
        fade_long = widening & (out["recent_return"] < -out["baseline_width"])
        fade_short = widening & (out["recent_return"] > out["baseline_width"])
        required = ["recent_lower", "recent_upper", "baseline_lower", "baseline_upper", "recent_return", "atr"]
        out["ready"] = np.isfinite(out[required]).all(axis=1) & (out["baseline_width"] > 0) & (out["atr"] > 0)
        out["candidate"] = np.select([upward | fade_long, downward | fade_short], [1, -1], default=0)
        out.loc[~out["ready"], "candidate"] = 0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QuantileMigrationParams) -> SignalFrame:
        if not np.isfinite(params.activation_pct) or params.activation_pct <= 0:
            raise ValueError("activation_pct must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = 0
        entry = 0.0
        entry_bar = -1
        armed = False
        stop = -np.inf
        for i in range(n):
            if pending:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError("Entry open must be finite and positive")
                side = pending
                pending = 0
                entry = opens[i]
                entry_bar = i
                armed = False
                stop = -np.inf
            if side:
                exit_now = i - entry_bar + 1 >= 18
                if np.isfinite(closes[i]) and closes[i] > 0:
                    favorable = side * (closes[i] - entry)
                    if armed and favorable <= stop:
                        exit_now = True
                    if favorable >= params.activation_pct * entry:
                        armed = True
                    if armed:
                        # Signed profit coordinates make the stop rise for both directions.
                        stop = max(stop, 0.0)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, favorable - params.trail_atr * atr[i])
                        if favorable <= stop:
                            exit_now = True
                if exit_now:
                    side = 0
                else:
                    decisions[i] = side
            elif candidates[i]:
                pending = int(candidates[i])
                decisions[i] = pending
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
