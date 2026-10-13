from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ConcentrationParams:
    concentration_window: int = 12
    rank_window: int = 63
    momentum_window: int = 3
    regime_percentile: float = 0.5
    entry_percentile: float = 0.75
    atr_window: int = 14
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[ConcentrationParams]):
    strategy_id = "gen_local_1791262629"

    @classmethod
    def params_type(cls) -> type[ConcentrationParams]:
        return ConcentrationParams

    @staticmethod
    def warmup_bars(params: ConcentrationParams) -> int:
        return int(max(
            params.concentration_window + params.rank_window,
            params.momentum_window + params.rank_window,
            params.atr_window + 1,
        ) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ConcentrationParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None)
        out["positive_return"] = out["return"].clip(lower=0)
        out["positive_return_squared"] = out["positive_return"].pow(2)
        out["gain_sum"] = out["positive_return"].rolling(p.concentration_window).sum()
        out["gain_squared_sum"] = out["positive_return_squared"].rolling(p.concentration_window).sum()
        out["concentration"] = out["gain_squared_sum"] / out["gain_sum"].pow(2).replace(0, np.nan)
        out["concentration_rank"] = out["concentration"].rolling(p.rank_window).rank(pct=True)
        out["momentum"] = close.pct_change(p.momentum_window, fill_method=None)
        out["momentum_rank"] = out["momentum"].rolling(p.rank_window).rank(pct=True)
        out["drift"] = close.pct_change(p.concentration_window, fill_method=None)
        out["previous_close"] = close.shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["daily_range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out[["concentration_rank", "momentum_rank", "drift", "return", "atr"]].notna().all(axis=1)
        valid &= out["atr"].gt(0) & close.gt(0)
        distributed = out["concentration_rank"].le(p.regime_percentile)
        concentrated = out["concentration_rank"].gt(p.regime_percentile)
        out["regime"] = np.select([valid & distributed, valid & concentrated], [1, 2], default=0).astype(int)
        continuation = distributed & out["momentum_rank"].ge(p.entry_percentile)
        digestion = concentrated & out["momentum_rank"].le(1.0 - p.entry_percentile)
        out["entry_ready"] = (
            valid & out["drift"].gt(0) & out["return"].gt(0)
            & (continuation | digestion)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ConcentrationParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        regime = indicators["regime"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if valid:
                    high_water = close[i] if not np.isfinite(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or age >= params.max_hold:
                    active = False
                else:
                    raw[i] = 1
            elif (
                valid and i > 0 and ready[i] and ready[i - 1]
                and regime[i] > 0 and regime[i] == regime[i - 1]
            ):
                active = True
                # The shifted entry fills next bar; initialize from that bar's close.
                high_water = np.nan
                stop = -np.inf
                age = 0
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    prices = pd.DataFrame({"close": [100., 101., 102., 105., 104., 101., 100.]}, index=index)
    derived = pd.DataFrame({
        "atr": [1., 1., 1., 1., 10., 10., 1.],
        "entry_ready": [True, True, False, False, False, False, False],
        "regime": [1, 1, 1, 1, 1, 1, 1],
    }, index=index)
    result = GeneratedStrategy().generate_signals(prices, derived, None, ConcentrationParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()
