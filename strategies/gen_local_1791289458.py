from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    path_bars: int = 5
    seasonal_bars: int = 252
    atr_bars: int = 14
    hold_days: int = 4
    area_threshold: float = 0.15
    seasonal_threshold: float = 0.12
    trail_atr: float = 1.8


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local_1791289458"

    @classmethod
    def params_type(cls):
        return SignatureSeasonParams

    def warmup_bars(self, params: SignatureSeasonParams) -> int:
        return int(max(params.seasonal_bars + params.hold_days + 1,
                       params.path_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        if (params.path_bars < 2 or params.seasonal_bars < 30
                or params.atr_bars < 1 or not 3 <= params.hold_days <= 5
                or params.trail_atr <= 0
                or not 0 <= params.area_threshold < 1
                or not 0 <= params.seasonal_threshold < 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        length = params.path_bars
        # Antisymmetric second-order signature of (elapsed time, cumulative return).
        weighted = sum((2 * j - length + 1) * out["return"].shift(length - 1 - j)
                       for j in range(length))
        out["path_area"] = weighted / (2.0 * length)
        out["path_variation"] = out["return"].abs().rolling(length).sum()
        out["path_strength"] = (
            weighted / ((length - 1) * out["path_variation"].replace(0, np.nan))
        ).clip(-1.0, 1.0)
        out["weekday"] = pd.Series(data.index.dayofweek, index=data.index)
        # Only completed holding-period outcomes known before today's close.
        out["mature_return"] = close.pct_change(params.hold_days, fill_method=None).shift(1)
        out["mature_weekday"] = out["weekday"].shift(params.hold_days + 1)
        out["seasonal_edge"] = np.nan
        out["seasonal_count"] = 0.0
        valid = out["mature_return"].notna()
        for day in range(5):
            member = out["mature_weekday"].eq(day)
            count = member.astype(float).where(valid).rolling(params.seasonal_bars).sum()
            total = out["mature_return"].where(member, 0.0).where(valid).rolling(
                params.seasonal_bars).sum()
            absolute = out["mature_return"].abs().where(member, 0.0).where(valid).rolling(
                params.seasonal_bars).sum()
            out[f"weekday_{day}_count"] = count
            out[f"weekday_{day}_mean"] = total / count.replace(0, np.nan)
            out[f"weekday_{day}_absolute_mean"] = absolute / count.replace(0, np.nan)
            current = out["weekday"].eq(day)
            out.loc[current, "seasonal_edge"] = (
                total / absolute.replace(0, np.nan)
            ).loc[current]
            out.loc[current, "seasonal_count"] = count.loc[current]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureSeasonParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.2}, index=data.index)
        strength = (indicators["path_strength"].clip(0, 1)
                    * indicators["seasonal_edge"].clip(0, 1)).fillna(0)
        entry_size = (0.2 + 0.75 * np.sqrt(strength)).clip(0.2, 0.95)
        eligible = (
            indicators["path_strength"].gt(params.area_threshold)
            & indicators["seasonal_edge"].gt(params.seasonal_threshold)
            & indicators["seasonal_count"].ge(20)
        ).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        held_size = 0.2
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0):
                active = False
                continue
            if active:
                # First iteration after an entry decision is the actual fill bar.
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or age >= params.hold_days:
                    active = False
            elif i >= self.warmup_bars(params) and eligible[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                held_size = float(entry_size.iloc[i])
            if active:
                signals[i] = 1
                sizes[i] = held_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SignatureSeasonParams(seasonal_bars=30)
    strategy = GeneratedStrategy()
    n = strategy.warmup_bars(params) + 9
    index = pd.bdate_range("2020-01-01", periods=n)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    entry = strategy.warmup_bars(params)
    features = pd.DataFrame({"path_strength": 0.0, "seasonal_edge": 0.5,
                             "seasonal_count": 25.0, "atr": 1.0}, index=index)
    features.iloc[entry, features.columns.get_loc("path_strength")] = 0.8
    ctx = StrategyContext("SPY", "1d", entry)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[entry:entry + 6].tolist() == [0, 1, 1, 1, 1, 0]
    assert result["size"].iloc[entry + 1] > 0.2
    data.iloc[entry + 1, data.columns.get_loc("close")] = 104.0
    data.iloc[entry + 2, data.columns.get_loc("close")] = 101.0
    features.iloc[entry + 2, features.columns.get_loc("atr")] = 3.0
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[entry + 1:entry + 4].tolist() == [1, 1, 0]
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    pd.testing.assert_frame_equal(strategy.indicators(data, params).iloc[:entry],
                                  strategy.indicators(data.iloc[:entry], params))


if __name__ == "__main__":
    _self_check()
