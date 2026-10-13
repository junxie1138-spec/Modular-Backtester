from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    area_bars: int = 5
    seasonal_bars: int = 252
    atr_bars: int = 14
    hold_bars: int = 4
    area_threshold: float = 0.025
    seasonal_floor: float = 0.002
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0

    def __post_init__(self):
        if self.area_bars < 2 or self.seasonal_bars < 100 or self.atr_bars < 2:
            raise ValueError("Invalid indicator windows")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        values = (self.area_threshold, self.seasonal_floor, self.breakeven_pct, self.trail_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Parameters must be finite")
        if min(self.area_threshold, self.breakeven_pct, self.trail_atr) <= 0:
            raise ValueError("Area threshold, breakeven trigger and ATR multiple must be positive")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791279182"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.area_bars + 1, params.atr_bars + 1,
                       params.seasonal_bars + params.hold_bars))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        previous = close.shift(1)
        out["gap"] = opening / previous - 1.0
        out["body"] = close / opening - 1.0
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()

        # Antisymmetric level-two signature of cumulative (gap, body) returns.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.area_bars):
            cross = out["gap"].shift(lag) * out["body"] - out["body"].shift(lag) * out["gap"]
            area = area + 0.5 * cross.rolling(params.area_bars - lag).sum()
        mass = (out["gap"].abs() + out["body"].abs()).rolling(params.area_bars).sum()
        out["area"] = area / mass.pow(2).where(mass > 0)
        out["gap_sum"] = out["gap"].rolling(params.area_bars).sum()
        out["body_sum"] = out["body"].rolling(params.area_bars).sum()

        # Only already-matured outcomes enter the originating weekday's estimate.
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        origin_weekday = weekday.shift(params.hold_bars)
        out["matured_return"] = close / close.shift(params.hold_bars) - 1.0
        out["seasonal"] = np.nan
        for day in range(5):
            sample = out["matured_return"].where(origin_weekday.eq(day))
            count = sample.notna().astype(float).rolling(params.seasonal_bars).sum()
            total = sample.fillna(0.0).rolling(params.seasonal_bars).sum()
            mean = total / count.where(count >= 20)
            mask = weekday.eq(day)
            out.loc[mask, "seasonal"] = mean.loc[mask]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (close > out["ma200"])
            & (out["atr"] > 0)
            & (out["area"] < -params.area_threshold)
            & (out["gap_sum"] < 0)
            & (out["body_sum"] > 0)
            & (out["seasonal"] > params.seasonal_floor)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        eligible = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if pending:
                pending = False
                # The entry reference becomes observable on the next bar.
                if i == 0 or not np.isfinite(opening) or opening <= 0 or not np.isfinite(atr[i - 1]):
                    continue
                active = True
                armed = False
                entry = peak = opening
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
            if active:
                age += 1
                valid = np.isfinite([high, low, close, atr[i], ma[i]]).all() and atr[i] > 0
                if not valid or low <= stop or close <= ma[i] or age >= params.hold_bars:
                    active = False
                else:
                    peak = max(peak, high)
                    armed = armed or high >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                        # Today's low is tested only against the previously known stop:
                        # the daily high/low order is unknown. The close follows the high.
                        if close <= stop:
                            active = False
                raw[i] = int(active)
            elif eligible[i]:
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = StrategyParams()
    s = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 100.4, "low": 99.6,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "ma200": 90.0, "entry": False}, index=index)
    ind.loc[index[[0, 7]], "entry"] = True
    data.loc[index[8], ["high", "close"]] = [104.0, 103.0]
    data.loc[index[9], ["open", "high", "low", "close"]] = [103.0, 103.0, 102.1, 102.5]
    ind.loc[index[9], "atr"] = 10.0
    data.loc[index[10], ["open", "high", "low", "close"]] = [102.5, 102.7, 101.9, 102.2]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = s.generate_signals(data, ind, ctx, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    data.loc[index[8], ["high", "close"]] = [101.2, 99.9]
    result = s.generate_signals(data, ind, ctx, p).data
    assert result.loc[index[9], "signal"] == 0
    assert (result["size"] > 0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1.0},
                        index=pd.bdate_range("2020-01-01", periods=300))
    actual = s.indicators(flat, p)
    assert not actual["entry"].any()
    assert s.warmup_bars(p) >= 256


if __name__ == "__main__":
    _self_check()
