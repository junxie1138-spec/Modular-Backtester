from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalAuctionParams:
    complexity_window: int = 30
    pressure_window: int = 18
    entropy_cut: float = 0.82
    entry_threshold: float = 0.12
    min_size: float = 0.15
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[OrdinalAuctionParams]):
    strategy_id = "gen_local-w2_1790999251"

    @classmethod
    def params_type(cls):
        return OrdinalAuctionParams

    def warmup_bars(self, params: OrdinalAuctionParams) -> int:
        return int(max(params.complexity_window + 2, params.pressure_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalAuctionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        volume = data["volume"].where(data["volume"] > 0)
        out["friction"] = ((data["high"] - data["low"]).clip(lower=0) / volume).replace([np.inf, -np.inf], np.nan)
        out["friction_lag1"] = out["friction"].shift(1)
        out["friction_lag2"] = out["friction"].shift(2)
        a, b, c = out["friction_lag2"], out["friction_lag1"], out["friction"]
        valid = a.notna() & b.notna() & c.notna()
        # Stable chronological ordering resolves equal observations deterministically.
        out["pattern"] = pd.Series(
            np.select(
                [(a <= b) & (b <= c), (a <= c) & (c < b),
                 (b < a) & (a <= c), (b <= c) & (c < a),
                 (c < a) & (a <= b)],
                [0, 1, 2, 3, 4], default=5,
            ), index=data.index, dtype=float,
        ).where(valid)
        out["complexity"] = 0.0
        for k in range(6):
            name = f"pattern_probability_{k}"
            out[name] = (out["pattern"] == k).astype(float).where(valid).rolling(
                params.complexity_window, min_periods=params.complexity_window
            ).mean()
            out["complexity"] -= out[name] * np.log(out[name].clip(lower=1e-12)) / np.log(6.0)

        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["overnight_return"] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["intraday_return"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        for session in ("overnight", "intraday"):
            returns = out[f"{session}_return"]
            out[f"{session}_net"] = returns.rolling(
                params.pressure_window, min_periods=params.pressure_window
            ).sum()
            out[f"{session}_travel"] = returns.abs().rolling(
                params.pressure_window, min_periods=params.pressure_window
            ).sum()
            denominator = out[f"{session}_travel"]
            out[f"{session}_pressure"] = (
                out[f"{session}_net"] / denominator.where(denominator > 0)
            ).where(denominator != 0, 0.0).clip(-1.0, 1.0)

        ready = out[["complexity", "overnight_pressure", "intraday_pressure"]].notna().all(axis=1)
        out["ordered_regime"] = (out["complexity"] < params.entropy_cut).astype(float).where(ready)
        # Ordered friction routes evidence to the continuous auction; disorder to opening gaps.
        out["score"] = out["intraday_pressure"].where(
            out["ordered_regime"] == 1.0, out["overnight_pressure"]
        ).where(ready)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalAuctionParams,
    ) -> SignalFrame:
        score = indicators["score"]
        desired = pd.Series(
            np.select(
                [score > params.entry_threshold, score < -params.entry_threshold],
                [1, -1], default=0,
            ), index=data.index, dtype=int,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(desired.to_numpy()):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Opposing entry evidence exits to flat; persistent evidence can enter next bar.
                position = 0
            # Neutral or missing evidence never exits an existing position.
            signals[i] = position

        conviction = ((score.abs() - params.entry_threshold) / max(
            1.0 - params.entry_threshold, 1e-12
        )).clip(0.0, 1.0).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = (params.min_size + (params.max_size - params.min_size) * conviction).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
