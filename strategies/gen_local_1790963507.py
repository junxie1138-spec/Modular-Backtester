from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelativeBrakeParams:
    rank_window: int = 40
    tail: float = 0.15
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[RelativeBrakeParams]):
    strategy_id = "gen_local_1790963507"

    @classmethod
    def params_type(cls):
        return RelativeBrakeParams

    @staticmethod
    def warmup_bars(params: RelativeBrakeParams) -> int:
        return int(max(params.rank_window + 2, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RelativeBrakeParams) -> pd.DataFrame:
        if not (
            params.rank_window >= 2
            and params.atr_window >= 1
            and 0.0 < params.tail < 0.35
            and np.isfinite(params.stop_atr)
            and params.stop_atr > 0.0
            and 3 <= params.hold_days <= 5
        ):
            raise ValueError("Invalid lookbacks, tail, stop multiplier, or holding period.")
        close = data["close"].where(data["close"] > 0.0)
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        rank = ret.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        previous_rank = rank.shift(1)
        previous_return = ret.shift(1)
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = (
            ret.notna() & previous_return.notna() & rank.notna()
            & previous_rank.notna() & atr.gt(0.0) & np.isfinite(atr)
        )
        central = rank.between(0.35, 0.65)
        buy = ready & central & previous_rank.le(params.tail) & previous_return.lt(0.0) & ret.lt(0.0)
        sell = ready & central & previous_rank.ge(1.0 - params.tail) & previous_return.gt(0.0) & ret.gt(0.0)
        strength = (
            (rank - previous_rank).abs() * (2.0 * (previous_rank - 0.5).abs()) / 0.5
        ).clip(0.0, 1.0).fillna(0.0)
        return pd.DataFrame(
            {
                "return": ret,
                "previous_return": previous_return,
                "return_rank": rank,
                "previous_rank": previous_rank,
                "true_range": true_range,
                "atr": atr,
                "ready": ready,
                "central": central,
                "candidate": buy.astype(int) - sell.astype(int),
                "strength": strength,
            },
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RelativeBrakeParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        extreme = np.nan
        stop = np.nan
        trade_size = 0.25
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0):
                position = 0
                extreme = stop = np.nan
                continue
            if position:
                # Entry fills on the bar after its decision; only held closes count.
                if position == 1:
                    extreme = close if np.isnan(extreme) else max(extreme, close)
                    proposed = extreme - params.stop_atr * atr
                    stop = proposed if np.isnan(stop) else max(stop, proposed)
                    stopped = close <= stop
                else:
                    # Short positions use the symmetric rolling-low trailing stop.
                    extreme = close if np.isnan(extreme) else min(extreme, close)
                    proposed = extreme + params.stop_atr * atr
                    stop = proposed if np.isnan(stop) else min(stop, proposed)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.hold_days:
                    position = 0
                    extreme = stop = np.nan
                    continue
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
                extreme = stop = np.nan
                trade_size = float(0.25 + 0.70 * strengths[i])
            signals[i] = position
            if position:
                sizes[i] = trade_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    for direction in (1, -1):
        closes = np.array([100.0, 103.0, 102.5, 101.5, 101.0])
        if direction == -1:
            closes = 200.0 - closes
        data = pd.DataFrame({"close": closes}, index=index)
        indicators = pd.DataFrame(
            {"atr": [1.0, 1.0, 0.5, 2.0, 1.0],
             "candidate": [direction, 0, 0, 0, 0], "strength": 0.5},
            index=index,
        )
        params = RelativeBrakeParams(stop_atr=2.0, hold_days=5)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert np.allclose(result["size"].iloc[1:4], 0.60)
        params = RelativeBrakeParams(stop_atr=100.0, hold_days=3)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]


if __name__ == "__main__":
    _self_check()
