from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 12
    regime_window: int = 30
    atr_window: int = 14
    innovation_threshold: float = 0.025
    forecast_edge: float = 0.01
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791269149"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.slope_window, params.atr_window) + params.regime_window + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.slope_window, params.regime_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if params.hold_bars < 1 or not np.isfinite(
            [params.innovation_threshold, params.forecast_edge, params.stop_atr]
        ).all() or min(
            params.innovation_threshold, params.forecast_edge, params.stop_atr
        ) <= 0:
            raise ValueError("Thresholds, stop distance and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().replace(0.0, np.nan)
        clock = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        n = params.slope_window
        out["slope"] = close.rolling(n, min_periods=n).cov(clock) / (n * (n + 1) / 12.0)
        out["innovation"] = out["slope"].diff() / out["atr"]
        out["change"] = close.diff()
        out["forecast_error"] = out["change"] - out["slope"].shift(1)
        out["forecast_gain"] = (
            (out["change"].pow(2) - out["forecast_error"].pow(2))
            / out["atr"].shift(1).pow(2)
        )
        out["regime_score"] = out["forecast_gain"].rolling(
            params.regime_window, min_periods=params.regime_window
        ).mean()
        out["regime"] = np.select(
            [out["regime_score"] > params.forecast_edge,
             out["regime_score"] < -params.forecast_edge],
            [1, -1],
            default=0,
        )
        out["candidate"] = (
            np.sign(out["innovation"]).fillna(0).astype(int) * out["regime"]
        ).where(out["innovation"].abs() >= params.innovation_threshold, 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(
            (candidate != 0) & candidate.eq(candidate.shift(1)), 0
        )
        # Only a fresh two-bar confirmation can start another trade.
        fresh = confirmed.where(confirmed.ne(confirmed.shift(1)), 0).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Entry executes at this open; ATR was frozen at the decision close.
                    fixed_stop = opens[i] - position * params.stop_atr * atr[i - 1]
                stopped = (
                    not np.isfinite(closes[i])
                    or not np.isfinite(fixed_stop)
                    or position * (closes[i] - fixed_stop) < 0
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
                signals[i] = position
                # An exit always produces a flat bar before another entry.
                continue
            if (
                fresh[i] != 0 and np.isfinite(closes[i])
                and np.isfinite(atr[i]) and atr[i] > 0
            ):
                position = int(fresh[i])
                entry_bar = i + 1
                fixed_stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = pd.DataFrame({"candidate": 1, "atr": 1.0}, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0] + [1] * 8 + [0, 0]
    indicators["candidate"] = [1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 97.0
    data.loc[index[6], "close"] = 103.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0]
    assert (result["size"] > 0).all()
    history = pd.concat([data] * 10, ignore_index=True)
    history.index = pd.date_range("2020-01-01", periods=len(history))
    baseline = strategy.indicators(history, params)
    altered = history.copy()
    altered.iloc[80:, altered.columns.get_loc("close")] += 50.0
    pd.testing.assert_frame_equal(
        baseline.iloc[:80], strategy.indicators(altered, params).iloc[:80]
    )


if __name__ == "__main__":
    _self_check()
