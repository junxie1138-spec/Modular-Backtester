from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    trend_bars: int = 8
    extreme_bars: int = 20
    atr_bars: int = 14
    min_drawdown_atr: float = 1.0
    min_revision: float = 0.12
    breakeven_pct: float = 0.005
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791200379"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.trend_bars + 2, params.extreme_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.trend_bars, params.extreme_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if min(params.min_revision, params.trail_atr) <= 0:
            raise ValueError("Revision threshold and ATR multiplier must be positive.")
        if min(params.min_drawdown_atr, params.breakeven_pct) < 0:
            raise ValueError("Drawdown and breakeven thresholds must be nonnegative.")
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean().replace(0, np.nan)
        n = params.trend_bars
        time = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        total = close.rolling(n, min_periods=n).sum()
        covariance = (time * close).rolling(n, min_periods=n).sum() - time.rolling(n, min_periods=n).mean() * total
        slope = covariance / (n * (n * n - 1) / 12.0)
        level = total / n + slope * ((n - 1) / 2.0)
        forecast = level.shift(1) + slope.shift(1)
        residual = (close - forecast) / atr
        peak = close.rolling(params.extreme_bars, min_periods=params.extreme_bars).max()
        trough = close.rolling(params.extreme_bars, min_periods=params.extreme_bars).min()
        return pd.DataFrame({
            "true_range": tr,
            "atr": atr,
            "time": time,
            "rolling_total": total,
            "covariance": covariance,
            "slope": slope,
            "level": level,
            "forecast": forecast,
            "residual": residual,
            "prior_residual": residual.shift(1),
            "prior_slope": slope.shift(1),
            "revision": slope.diff() / atr,
            "peak": peak,
            "trough": trough,
            "drawdown": (peak - close) / atr,
            "drawup": (close - trough) / atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        long_entry = ready & (indicators["drawdown"] >= params.min_drawdown_atr) & (indicators["prior_slope"] < 0) & (indicators["revision"] >= params.min_revision) & (indicators["prior_residual"] <= 0) & (indicators["residual"] > 0)
        short_entry = ready & (indicators["drawup"] >= params.min_drawdown_atr) & (indicators["prior_slope"] > 0) & (indicators["revision"] <= -params.min_revision) & (indicators["prior_residual"] >= 0) & (indicators["residual"] < 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        revision = indicators["revision"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        side, age = 0, 0
        entry = stop = best = np.nan
        armed = False
        position_size = 0.25
        for i in range(len(data)):
            if side:
                if not np.isfinite(prices[i]).all() or not np.isfinite(atr[i]):
                    side = 0
                    continue
                # Entry is the open following the decision, never the decision close.
                if age == 0:
                    entry = side * prices[i, 0]
                    best = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
                age += 1
                favorable = prices[i, 1] if side == 1 else -prices[i, 2]
                best = max(best, favorable)
                # Signed price makes the same upward-only ratchet work for shorts.
                if best - entry >= abs(entry) * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, entry, best - params.trail_atr * atr[i])
                # Stops are close decisions, executed at the following open.
                if side * prices[i, 3] <= stop or age >= 2:
                    side = 0
                    age = 0
                    continue
                signal[i] = side
                sizes[i] = position_size
            elif i >= self.warmup_bars(params) - 1 and candidates[i]:
                if not np.isfinite(prices[i]).all() or prices[i, 3] <= 0:
                    continue
                side = int(candidates[i])
                age = 0
                position_size = float(np.clip(0.4 + 0.25 * abs(revision[i]) / params.min_revision, 0.25, 0.95))
                signal[i] = side
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RecoveryParams(trend_bars=2, extreme_bars=2, atr_bars=2)
    index = pd.date_range("2020-01-01", periods=8)
    for side in (1, -1):
        # A gap at entry distinguishes actual fill price from decision price.
        prices = np.full((8, 4), 100.0)
        prices[4] = [102.0, 103.0, 101.0, 102.8]
        prices[5] = [102.8, 102.9, 102.0, 102.2]
        if side == -1:
            prices = 200.0 - prices
            prices[:, [1, 2]] = prices[:, [2, 1]]
        data = pd.DataFrame(prices, columns=["open", "high", "low", "close"], index=index)
        data["volume"] = 1000.0
        ind = pd.DataFrame({
            "atr": 1.0, "drawdown": 2.0, "drawup": 2.0,
            "prior_slope": -float(side), "revision": 0.0,
            "prior_residual": -float(side), "residual": float(side),
        }, index=index)
        ind.loc[index[3], "revision"] = side * 0.24
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, 0, 0, 0, side, side, 0, 0]
        assert (result["size"] > 0).all()
        actual = strategy.indicators(data, params)
        prefix = strategy.indicators(data.iloc[:6], params)
        pd.testing.assert_frame_equal(actual.iloc[:6], prefix)
    # A close beyond the armed breakeven stop must flatten after one session.
    data.loc[index[4], ["high", "low", "close"]] = [99.0, 96.0, 98.1]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[4] == -1 and result["signal"].iloc[5] == 0


if __name__ == "__main__":
    _self_check()
