from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    season_lookback: int = 252
    dip_bars: int = 3
    dip_threshold: float = 0.005
    atr_bars: int = 14
    atr_mult: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local-w2_1791239578"

    @classmethod
    def params_type(cls):
        return SeasonalParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.season_lookback + 1,
                       params.dip_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data, params):
        if (params.season_lookback < 30 or params.dip_bars < 1
                or params.atr_bars < 1 or not 3 <= params.hold_bars <= 5
                or not np.isfinite(params.atr_mult) or params.atr_mult <= 0
                or not np.isfinite(params.dip_threshold) or params.dip_threshold <= 0):
            raise ValueError("Invalid seasonal strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["early_month"] = data.index.day <= 7
        out["dip"] = data["close"].pct_change(
            periods=params.dip_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        valid = out["return"].notna()
        early = out["early_month"] & valid
        other = ~out["early_month"] & valid
        window = params.season_lookback
        early_count = early.astype(float).rolling(window, min_periods=window).sum()
        other_count = other.astype(float).rolling(window, min_periods=window).sum()
        out["early_mean"] = out["return"].where(early, 0.0).rolling(
            window, min_periods=window).sum() / early_count.where(early_count >= 10)
        out["other_mean"] = out["return"].where(other, 0.0).rolling(
            window, min_periods=window).sum() / other_count.where(other_count >= 10)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars).mean()
        out["entry"] = (
            out["early_month"]
            & (out["dip"] <= -params.dip_threshold)
            & (out["early_mean"] > 0)
            & (out["early_mean"] > out["other_mean"])
            & (data["close"] > out["ma200"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        n = len(data)
        raw = np.zeros(n, dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        for i in range(n):
            if entry_bar >= 0:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        entry_bar = -1
                        continue
                    # Freeze decision-bar ATR against the next-bar entry open.
                    stop = opens[i] - params.atr_mult * entry_atr
                age = i - entry_bar + 1
                if (not np.isfinite(close[i]) or not np.isfinite(regime[i])
                        or close[i] < stop or close[i] <= regime[i]
                        or age >= params.hold_bars):
                    entry_bar = -1
                    continue
                raw[i] = 1
            elif (i + 1 < n and entries[i] and np.isfinite(close[i])
                    and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(regime[i]) and close[i] > regime[i]):
                raw[i] = 1
                entry_bar = i + 1
                entry_atr = atr[i]
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "ma200": 90.0,
                            "entry": [True] + [False] * 7}, index=index)
    params = SeasonalParams(hold_bars=3)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    data.iloc[1, data.columns.get_loc("open")] = 101.0
    data.iloc[2, data.columns.get_loc("close")] = 98.5
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert result["size"].gt(0).all()
    derived["entry"] = False
    assert GeneratedStrategy.generate_signals(
        data, derived, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
