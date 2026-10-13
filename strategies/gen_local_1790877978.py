from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FeedbackParams:
    roc_days: int = 3
    feedback_window: int = 20
    atr_days: int = 14
    acceleration_threshold: float = 0.5
    regime_margin: float = 0.1
    trail_atr: float = 1.6
    hold_days: int = 4
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[FeedbackParams]):
    strategy_id = "gen_local_1790877978"

    @classmethod
    def params_type(cls):
        return FeedbackParams

    def warmup_bars(self, params: FeedbackParams) -> int:
        return int(max(params.roc_days + params.feedback_window + 2, params.atr_days + 1))

    def indicators(self, data: pd.DataFrame, params: FeedbackParams) -> pd.DataFrame:
        if (params.roc_days < 1 or params.feedback_window < 2 or params.atr_days < 1
                or not 3 <= params.hold_days <= 5
                or not 0 < params.minimum_size <= params.maximum_size <= 1
                or params.acceleration_threshold <= 0 or params.trail_atr <= 0
                or not 0 <= params.regime_margin < 2):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["roc"] = close.pct_change(params.roc_days, fill_method=None)
        out["activity"] = volume.pct_change(fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_scale"] = out["acceleration"].rolling(
            params.feedback_window
        ).std(ddof=0).shift(1)
        out["score"] = out["acceleration"] / out["acceleration_scale"].where(
            out["acceleration_scale"] > 1e-12
        )
        out["volume_leads"] = out["roc"].rolling(params.feedback_window).corr(
            out["activity"].shift(1)
        ).clip(-1, 1)
        out["price_leads"] = out["activity"].rolling(params.feedback_window).corr(
            out["roc"].shift(1)
        ).clip(-1, 1)
        out["feedback"] = out["volume_leads"] - out["price_leads"]
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_days).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FeedbackParams,
    ) -> SignalFrame:
        score = indicators["score"]
        regime_ok = (
            ((indicators["feedback"] > params.regime_margin) & (indicators["roc"] > 0))
            | ((indicators["feedback"] < -params.regime_margin) & (indicators["roc"] < 0))
        )
        ready = indicators[["score", "feedback", "roc", "atr"]].notna().all(axis=1)
        entry = (
            ready & regime_ok & (indicators["atr"] > 0)
            & (score > params.acceleration_threshold)
            & (score.shift(1) <= params.acceleration_threshold)
        ).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        strengths = score.to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        active = False
        entered = -1
        highest = -np.inf
        stop = -np.inf
        exposure = params.minimum_size
        for i in range(len(data)):
            if active:
                # The entry fills this bar; seed the high from its close.
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                else:
                    highest = max(highest, closes[i])
                    stop = max(stop, highest - params.trail_atr * atr[i])
                    if closes[i] <= stop or i - entered >= params.hold_days:
                        active = False
            elif i >= self.warmup_bars(params) and entry[i] and np.isfinite(closes[i]):
                active = True
                entered = i
                highest = -np.inf
                stop = -np.inf
                strength = max(0.0, strengths[i])
                exposure = params.minimum_size + (
                    params.maximum_size - params.minimum_size
                ) * strength / (1.0 + strength)
            if active:
                signals[i] = 1
                sizes[i] = exposure
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = FeedbackParams(roc_days=1, feedback_window=2, atr_days=1)
    index = pd.date_range("2020-01-01", periods=12)
    close = np.full(12, 100.0)
    close[6] = 102.0
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(12, 1000.0),
    }, index=index)
    ind = pd.DataFrame({"score": 0.0, "feedback": 0.5, "roc": 0.01, "atr": 1.0}, index=index)
    ind.loc[index[5], "score"] = 1.0
    ind.loc[index[7], "atr"] = 10.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert params.minimum_size < result["size"].iloc[6] < params.maximum_size
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
