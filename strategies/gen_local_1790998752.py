from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DepthParams:
    depth_atr: float = 2.5
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[DepthParams]):
    strategy_id = "gen_local_1790998752"

    @classmethod
    def params_type(cls) -> type[DepthParams]:
        return DepthParams

    def warmup_bars(self, params: DepthParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: DepthParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["atr"] = out["atr"].where(out["atr"] > 0)
        out["prior_peak"] = data["close"].shift(1).rolling(10, min_periods=10).max()
        out["prior_trough"] = data["close"].shift(1).rolling(10, min_periods=10).min()
        out["density"] = data["volume"].where(data["volume"] > 0) / out["range"]
        out["density_baseline"] = out["density"].shift(1).rolling(20, min_periods=20).median()
        out["drawdown_depth"] = (out["prior_peak"] - data["close"]) / out["atr"]
        out["drawup_depth"] = (data["close"] - out["prior_trough"]) / out["atr"]
        depleted = out["density"] < 0.65 * out["density_baseline"]
        ready = np.isfinite(out[["atr", "density", "density_baseline", "drawdown_depth", "drawup_depth"]]).all(axis=1)
        long_entry = ready & depleted & (out["drawdown_depth"] >= params.depth_atr) & (data["close"] < out["previous_close"])
        short_entry = ready & depleted & (out["drawup_depth"] >= params.depth_atr) & (data["close"] > out["previous_close"])
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DepthParams) -> SignalFrame:
        if not np.isfinite(params.depth_atr) or params.depth_atr <= 0:
            raise ValueError("depth_atr must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        stop_distance = 0.0
        fixed_stop = np.nan
        sessions = 0
        for i in range(len(data)):
            if position != 0:
                if pending_entry:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        pending_entry = False
                        continue
                    fixed_stop = opens[i] - position * stop_distance
                    pending_entry = False
                sessions += 1
                invalid_close = not np.isfinite(closes[i]) or closes[i] <= 0
                stopped = (position == 1 and closes[i] < fixed_stop) or (position == -1 and closes[i] > fixed_stop)
                if invalid_close or stopped or sessions >= 2:
                    position = 0
                    sessions = 0
            elif candidates[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(candidates[i])
                stop_distance = params.stop_atr * atr[i]
                pending_entry = True
                sessions = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
