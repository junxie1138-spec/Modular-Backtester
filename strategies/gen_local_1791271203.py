from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRecoveryParams:
    path_window: int = 8
    extreme_window: int = 42
    percentile_window: int = 126
    volume_window: int = 20
    atr_window: int = 14
    area_quantile: float = 0.75
    extreme_quantile: float = 0.25
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    hold_bars: int = 18
    cooldown_bars: int = 3


class GeneratedStrategy(BaseStrategy[SignatureRecoveryParams]):
    strategy_id = "gen_local_1791271203"

    @classmethod
    def params_type(cls):
        return SignatureRecoveryParams

    def warmup_bars(self, params: SignatureRecoveryParams) -> int:
        return int(max(
            params.volume_window + params.path_window + params.percentile_window,
            params.extreme_window + params.percentile_window,
            params.atr_window + 1,
            4,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureRecoveryParams) -> pd.DataFrame:
        p = params
        if min(p.path_window, p.extreme_window, p.percentile_window,
               p.volume_window, p.atr_window, p.hold_bars) < 2:
            raise ValueError("Lookbacks and holding period must be at least two")
        if not (0.5 < p.area_quantile < 1 and 0 < p.extreme_quantile < 0.5):
            raise ValueError("Invalid percentile parameters")
        if p.trail_atr <= 0 or p.breakeven_pct <= 0 or p.cooldown_bars < 0:
            raise ValueError("Invalid exit parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        out["volume_mean"] = data["volume"].where(data["volume"] >= 0).rolling(p.volume_window).mean()
        out["relative_volume"] = data["volume"].where(data["volume"] >= 0) / out["volume_mean"].replace(0, np.nan)
        spread = data["high"] - data["low"]
        out["settlement"] = ((2 * close - data["high"] - data["low"]) / spread.replace(0, np.nan)).where(spread != 0, 0.0).clip(-1, 1)
        out["pressure"] = out["relative_volume"] * out["settlement"]
        x, y = out["log_return"], out["pressure"]
        area = pd.Series(0.0, index=data.index)
        # Second-order antisymmetric signature: half the cross-product sum
        # over all ordered pairs of increments within the local path.
        for lag in range(1, p.path_window):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + cross.rolling(p.path_window - lag).sum()
        out["area_sum"] = 0.5 * area
        out["path_scale"] = x.abs().rolling(p.path_window).sum() * y.abs().rolling(p.path_window).sum()
        out["area"] = out["area_sum"] / out["path_scale"].replace(0, np.nan)
        out["area_threshold"] = out["area"].shift(1).rolling(p.percentile_window).quantile(p.area_quantile)
        out["drawdown"] = close / close.rolling(p.extreme_window).max() - 1
        out["drawup"] = close / close.rolling(p.extreme_window).min() - 1
        out["down_threshold"] = out["drawdown"].shift(1).rolling(p.percentile_window).quantile(p.extreme_quantile)
        out["up_threshold"] = out["drawup"].shift(1).rolling(p.percentile_window).quantile(1 - p.extreme_quantile)
        out["return_three"] = x.rolling(3).sum()
        out["pressure_three"] = y.rolling(3).sum()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            spread, (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureRecoveryParams) -> SignalFrame:
        p, ind = params, indicators
        needed = ["area", "area_threshold", "drawdown", "drawup",
                  "down_threshold", "up_threshold", "return_three",
                  "pressure_three", "atr"]
        ready = pd.Series(np.isfinite(ind[needed].to_numpy(dtype=float)).all(axis=1), index=data.index)
        geometry = ready & (ind["area"] > ind["area_threshold"])
        buy = geometry & (ind["drawdown"] < 0) & (ind["drawdown"] <= ind["down_threshold"]) & (ind["return_three"] > 0) & (ind["pressure_three"] > 0)
        sell = geometry & (ind["drawup"] > 0) & (ind["drawup"] >= ind["up_threshold"]) & (ind["return_three"] < 0) & (ind["pressure_three"] < 0)
        setups = np.where(buy, 1, np.where(sell, -1, 0))
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = ind["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position, entry_bar, next_entry = 0, -1, 0
        entry, stop = np.nan, np.nan
        armed = False
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if position:
                # Yesterday's target starts trading at today's open.
                if np.isnan(entry):
                    entry = opens[i]
                    entry_bar = i
                    stop = entry - position * p.trail_atr * atrs[i - 1]
                valid = np.isfinite([entry, stop, close, atr]).all() and entry > 0 and close > 0 and atr > 0
                # Close-confirmed exits are executed at the following open.
                stopped = valid and position * (close - stop) <= 0
                expired = i - entry_bar + 1 >= p.hold_bars
                if not valid or stopped or expired:
                    position = 0
                    entry = np.nan
                    armed = False
                    next_entry = i + p.cooldown_bars + 1
                else:
                    if position * (close / entry - 1) >= p.breakeven_pct:
                        armed = True
                    if armed:
                        candidate = close - position * p.trail_atr * atr
                        if position == 1:
                            stop = max(stop, entry, candidate)
                        else:
                            stop = min(stop, entry, candidate)
            elif i >= next_entry and setups[i] and np.isfinite([close, atr]).all() and close > 0 and atr > 0:
                position = int(setups[i])
                entry = np.nan
                armed = False
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    p = SignatureRecoveryParams(path_window=4, extreme_window=4,
                                percentile_window=5, volume_window=3, atr_window=3)
    index = pd.date_range("2020-01-01", periods=30)
    close = 100 + np.arange(30) * 0.1 + np.sin(np.arange(30))
    data = pd.DataFrame({"open": close, "high": close + 1.2,
                         "low": close - 0.8, "close": close,
                         "volume": 1000 + np.arange(30) * 10}, index=index)
    ind = GeneratedStrategy.indicators(None, data, p)
    x = ind["log_return"].iloc[-p.path_window:].to_numpy()
    y = ind["pressure"].iloc[-p.path_window:].to_numpy()
    expected = 0.5 * sum(x[a] * y[b] - y[a] * x[b]
                         for a in range(p.path_window) for b in range(a + 1, p.path_window))
    assert np.isclose(ind["area_sum"].iloc[-1], expected)
    # Both directions activate breakeven, ratchet, and exit on reversal.
    for direction in (1, -1):
        price = 100 + direction * np.array([0, 0, 3, 5, 4, 0, 0], dtype=float)
        idx = pd.date_range("2021-01-01", periods=len(price))
        bars = pd.DataFrame({"open": 100.0, "high": price + 1,
                             "low": price - 1, "close": price, "volume": 1000.0}, index=idx)
        forced = pd.DataFrame({"area": 1.0, "area_threshold": 0.5,
                               "drawdown": 0.0, "down_threshold": -0.05,
                               "drawup": 0.0, "up_threshold": 0.05,
                               "return_three": float(direction),
                               "pressure_three": float(direction), "atr": 1.0}, index=idx)
        forced.loc[idx[0], "drawdown" if direction == 1 else "drawup"] = -0.1 if direction == 1 else 0.1
        result = GeneratedStrategy.generate_signals(None, bars, forced, None, SignatureRecoveryParams()).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, direction, 0]
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
