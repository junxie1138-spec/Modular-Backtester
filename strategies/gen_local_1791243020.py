from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_order: float = 0.4
    fd_window: int = 32
    normalization_window: int = 40
    seasonal_window: int = 504
    atr_window: int = 14
    entry_threshold: float = 1.25
    hold_bars: int = 18
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if not (0.0 < self.fd_order < 1.0):
            raise ValueError("fd_order must be between zero and one")
        if min(self.fd_window, self.normalization_window, self.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two")
        if self.seasonal_window < 252 or not 15 <= self.hold_bars <= 20:
            raise ValueError("Require a yearly seasonal sample and a 15-20 session hold")
        if not (self.entry_threshold > 0 and self.breakeven_pct > 0 and self.trail_atr > 0):
            raise ValueError("Thresholds must be positive")
        if not (0 < self.min_size <= self.max_size <= 1):
            raise ValueError("Position sizes must be in (0, 1]")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791243020"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fd_window + params.normalization_window + params.seasonal_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        # ponytail: finite fractional memory; increase fd_window if truncation matters.
        weights = np.ones(params.fd_window, dtype=float)
        for k in range(1, params.fd_window):
            weights[k] = -weights[k - 1] * (params.fd_order - k + 1) / k
        fractional = close * weights[0]
        for k in range(1, params.fd_window):
            fractional = fractional + weights[k] * close.shift(k)
        out["fractional_close"] = fractional
        out["impulse"] = fractional.diff()
        out["impulse_mean"] = out["impulse"].rolling(params.normalization_window).mean()
        out["impulse_std"] = out["impulse"].rolling(params.normalization_window).std(ddof=0)
        out["impulse_z"] = ((out["impulse"] - out["impulse_mean"]) /
                            out["impulse_std"].replace(0.0, np.nan))
        out["seasonal_mean"] = np.nan
        out["seasonal_count"] = 0.0
        months = data.index.month
        for month in range(1, 13):
            mask = months == month
            valid = out["impulse_z"].notna() & mask
            count = valid.astype(float).rolling(params.seasonal_window, min_periods=1).sum().shift(1)
            total = out["impulse_z"].where(valid, 0.0).rolling(
                params.seasonal_window, min_periods=1).sum().shift(1)
            out.loc[mask, "seasonal_mean"] = (total / count.replace(0.0, np.nan)).loc[mask]
            out.loc[mask, "seasonal_count"] = count.loc[mask]
        out["anomaly"] = (out["impulse_z"] - out["seasonal_mean"]).replace(
            [np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        periods = pd.Series(data.index.to_period("M"), index=data.index)
        out["month_open"] = periods.ne(periods.shift(1))
        if len(out):
            out.iloc[0, out.columns.get_loc("month_open")] = False
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int),
                           "size": np.full(len(data), params.min_size)}, index=data.index)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        anomalies = indicators["anomaly"].to_numpy(dtype=float)
        counts = indicators["seasonal_count"].to_numpy(dtype=float)
        month_open = indicators["month_open"].to_numpy(dtype=bool)
        signal = df["signal"].to_numpy(copy=True)
        size = df["size"].to_numpy(copy=True)
        side, age = 0, 0
        pending, armed = False, False
        entry, best, stop = 0.0, 0.0, 0.0
        position_size = params.min_size
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            exited = False
            if side:
                if pending:
                    # Prior close's decision fills at this bar's open.
                    if not (np.isfinite(opens[i]) and opens[i] > 0):
                        side = 0
                        pending = False
                        exited = True
                    else:
                        entry = opens[i]
                        best = side * entry
                        stop = best - params.trail_atr * atrs[i - 1]
                        pending = False
                        armed = False
                        age = 0
                if side:
                    age += 1
                    price = closes[i]
                    signed_price = side * price
                    if not (np.isfinite(price) and price > 0 and np.isfinite(atrs[i]) and atrs[i] > 0):
                        side = 0
                        exited = True
                    elif signed_price <= stop or age >= params.hold_bars:
                        side = 0
                        exited = True
                    else:
                        best = max(best, signed_price)
                        if side * (price / entry - 1.0) >= params.breakeven_pct:
                            armed = True
                        if armed:
                            # Signed-price stops only rise; a short's price stop only falls.
                            stop = max(stop, side * entry, best - params.trail_atr * atrs[i])
            if not side and not exited and i >= warmup - 1:
                anomaly = anomalies[i]
                if (month_open[i] and counts[i] >= 20 and np.isfinite(anomaly) and
                        np.isfinite(closes[i]) and closes[i] > 0 and
                        np.isfinite(atrs[i]) and atrs[i] > 0 and
                        abs(anomaly) >= params.entry_threshold):
                    side = -1 if anomaly > 0 else 1
                    strength = np.clip((abs(anomaly) - params.entry_threshold) /
                                       (2.0 * params.entry_threshold), 0.0, 1.0)
                    position_size = float(params.min_size +
                                          (params.max_size - params.min_size) * strength)
                    pending = True
            signal[i] = side
            size[i] = position_size if side else params.min_size
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(fd_window=2, normalization_window=2, seasonal_window=252, atr_window=2)
    start = strategy.warmup_bars(params) - 1
    index = pd.bdate_range("2018-01-01", periods=start + 55)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "anomaly": np.nan,
                        "seasonal_count": 25.0, "month_open": False}, index=index)
    for offset, anomaly in [(0, -3.0), (12, 3.0), (25, -2.0)]:
        ind.iloc[start + offset, ind.columns.get_loc("anomaly")] = anomaly
        ind.iloc[start + offset, ind.columns.get_loc("month_open")] = True
    for offset, close in [(2, 103.0), (3, 102.0), (4, 99.0),
                          (14, 97.0), (15, 98.0), (16, 101.0)]:
        data.iloc[start + offset, data.columns.get_loc("close")] = close
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=start + 1)
    result = strategy.generate_signals(data, ind, ctx, params).data
    signals = result["signal"].to_numpy()
    assert signals[start] == 0 and signals[start + 1] == 1
    assert signals[start + 4] == 1 and signals[start + 5] == 0
    assert signals[start + 12] == 0 and signals[start + 13] == -1
    assert signals[start + 16] == -1 and signals[start + 17] == 0
    assert signals[start + 43] == 1 and signals[start + 44] == 0
    assert result["size"].between(params.min_size, params.max_size).all()
    assert result["size"].iloc[start + 1] > result["size"].iloc[start + 26]
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-10], params)
    pd.testing.assert_frame_equal(full.iloc[:-10], prefix)
    truncated = strategy.generate_signals(data.iloc[:-10], ind.iloc[:-10], ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:-10], truncated)


if __name__ == "__main__":
    _self_check()
