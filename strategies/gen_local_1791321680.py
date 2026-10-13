from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 12
    autocorr_window: int = 20
    score_threshold: float = 0.08
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1791321680"

    @classmethod
    def params_type(cls) -> type[TopologyParams]:
        return TopologyParams

    @staticmethod
    def warmup_bars(params: TopologyParams) -> int:
        if params.graph_window < 2 or params.autocorr_window < 3 or params.atr_window < 1:
            raise ValueError("Lookbacks are too short")
        if not 3 <= params.hold_days <= 5:
            raise ValueError("hold_days must be between 3 and 5")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        if not np.isfinite(params.score_threshold) or params.score_threshold <= 0:
            raise ValueError("score_threshold must be finite and positive")
        return int(max(params.graph_window + params.autocorr_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        upper = pd.Series(0.0, index=data.index)
        lower = pd.Series(0.0, index=data.index)
        interior_max = returns.shift(1)
        interior_min = returns.shift(1)
        # Endpoint degrees in horizontal visibility graphs of returns and negated returns.
        # Adjacent edges cancel in the signed degree difference.
        for lag in range(2, params.graph_window + 1):
            past = returns.shift(lag)
            upper += ((returns > interior_max) & (past > interior_max)).astype(float)
            lower += ((returns < interior_min) & (past < interior_min)).astype(float)
            interior_max = pd.concat([interior_max, past], axis=1).max(axis=1)
            interior_min = pd.concat([interior_min, past], axis=1).min(axis=1)
        complete = returns.rolling(params.graph_window + 1).count().eq(params.graph_window + 1)
        upper = upper.where(complete)
        lower = lower.where(complete)
        topology = (upper - lower) / (params.graph_window - 1)
        autocorr = topology.rolling(params.autocorr_window, min_periods=params.autocorr_window).corr(topology.shift(1))
        autocorr = autocorr.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        score = topology * autocorr
        candidate = pd.Series(0, index=data.index, dtype=int)
        ready = score.notna() & atr.gt(0)
        candidate.loc[ready & score.ge(params.score_threshold)] = 1
        candidate.loc[ready & score.le(-params.score_threshold)] = -1
        return pd.DataFrame({
            "returns": returns,
            "upper_degree": upper,
            "lower_degree": lower,
            "topology": topology,
            "topology_autocorr": autocorr,
            "true_range": true_range,
            "atr": atr,
            "score": score,
            "candidate": candidate,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high = np.nan
        low = np.nan
        trailing = -np.inf
        short_trailing = np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                targets[i] = 0
                continue
            if position:
                # First update is on the entry fill bar, never on the decision bar.
                age += 1
                high = close if np.isnan(high) else max(high, close)
                low = close if np.isnan(low) else min(low, close)
                trailing = max(trailing, high - params.stop_atr * atr)
                short_trailing = min(short_trailing, low + params.stop_atr * atr)
                # Required high-water stop applies to both sides.
                # Shorts also have a mirrored adverse-price stop.
                stopped = close <= trailing or (position == -1 and close >= short_trailing)
                if stopped or age >= params.hold_days:
                    position = 0
                    targets[i] = 0
                    continue
            elif confirmed[i] != 0:
                position = int(confirmed[i])
                age = 0
                high = low = np.nan
                trailing = -np.inf
                short_trailing = np.inf
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100., 100., 100., 103., 106., 102., 101., 101., 101.])
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.,
    }, index=index)
    supplied = pd.DataFrame({
        "candidate": [1, 1, 0, 0, 0, 0, 0, 0, 0],
        "atr": 1.0,
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, supplied, None, TopologyParams(hold_days=5)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
    supplied["candidate"] = [-1, -1, 0, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, supplied, None, TopologyParams()).data
    assert result["signal"].tolist() == [0, 0, -1, -1, 0, 0, 0, 0, 0]
    flat = data.assign(open=100., high=101., low=99., close=100.)
    derived = strategy.indicators(flat, TopologyParams())
    assert derived["candidate"].eq(0).all()
    supplied["candidate"] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(flat, supplied, None, TopologyParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
