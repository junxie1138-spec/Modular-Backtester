from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak_days: int = 3
    correlation_window: int = 40
    reversal_threshold: float = 0.15
    atr_period: int = 14
    trail_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791327191"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.correlation_window + 2, params.atr_period + 1, params.streak_days + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (params.streak_days >= 1 and params.correlation_window >= 3 and
                params.atr_period >= 1 and 3 <= params.hold_days <= 5 and
                np.isfinite(params.trail_atr) and params.trail_atr > 0 and
                np.isfinite(params.reversal_threshold) and 0 < params.reversal_threshold < 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap_return"] = (data["open"] / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        out["intraday_return"] = (data["close"] / data["open"].where(lambda s: s > 0) - 1).replace([np.inf, -np.inf], np.nan)
        negative_gap = out["gap_return"].lt(0)
        out["negative_gap_streak"] = negative_gap.astype(int).groupby((~negative_gap).cumsum()).cumsum()
        out["lagged_gap_return"] = out["gap_return"].shift(1)
        out["echo_correlation"] = out["lagged_gap_return"].rolling(params.correlation_window).corr(out["intraday_return"]).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        # Two primitives: consecutive negative gaps AND a reversing cross-session echo.
        out["entry"] = (out["negative_gap_streak"].ge(params.streak_days) &
                        out["echo_correlation"].le(-params.reversal_threshold) &
                        out["atr"].gt(0) & np.isfinite(out["atr"]) &
                        data["close"].gt(0) & np.isfinite(data["close"]))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if active:
                held += 1
                close, atr = closes[i], atrs[i]
                if not np.isfinite(close) or close <= 0:
                    active = False
                else:
                    # Entry fills on this bar's open; start tracking actual held closes.
                    high_water = max(high_water, close)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or held >= params.hold_days:
                        active = False
            elif entries[i]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
            raw[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 100., 103., 100., 100., 100., 100.]}, index=index)
    derived = pd.DataFrame({"atr": [1., 1., 1., 10., 10., 10., 10.],
                            "entry": [True, False, False, False, False, False, False]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data["close"] = np.arange(100., 107.)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
