from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    season_window: int = 126
    atr_window: int = 14
    expansion: float = 1.4
    contraction: float = 0.8
    stop_atr: float = 1.0
    hold_bars: int = 2
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790885709"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.season_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.season_window < 15 or params.atr_window < 1
                or params.expansion <= 1 or not 0 < params.contraction < 1
                or params.stop_atr <= 0 or params.hold_bars not in (1, 2)
                or not 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["relative_range"] = out["range"] / data["close"].where(data["close"] > 0)
        weekday = data.index.dayofweek
        out["weekday_range"] = np.nan
        for day in range(5):
            history = out["relative_range"].where(weekday == day).shift(1)
            baseline = history.rolling(
                params.season_window,
                min_periods=max(3, params.season_window // 14),
            ).mean()
            out.loc[weekday == day, "weekday_range"] = baseline.loc[weekday == day]
        out["range_ratio"] = out["relative_range"] / out["weekday_range"].where(out["weekday_range"] > 0)
        out["prior_ratio"] = out["range_ratio"].shift(1)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["close_location"] = ((data["close"] - data["low"]) / out["range"].where(out["range"] > 0)).clip(0, 1)
        out["prior_bearish"] = (data["close"] < data["open"]).shift(1, fill_value=False)
        out["strength"] = (
            (out["prior_ratio"] / params.expansion - 1).clip(lower=0)
            + (1 - out["range_ratio"]).clip(lower=0)
        ).mul(out["close_location"]).clip(0, 1).fillna(0)
        out["entry"] = (
            (out["prior_ratio"] >= params.expansion)
            & (out["range_ratio"] <= params.contraction)
            & out["prior_bearish"]
            & (out["close_location"] >= 0.65)
            & (data["close"] > data["open"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0).to_numpy(dtype=float)
        entry_bar = -1
        stop = np.nan
        exposure = params.min_size
        for i in range(self.warmup_bars(params), n):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Freeze the stop at the next-session entry open and decision-bar ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                if (not np.isfinite(close[i]) or not np.isfinite(stop)
                        or close[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                    continue
                signal[i] = 1
                sizes[i] = exposure
            elif eligible[i] and np.isfinite(atr[i]) and np.isfinite(close[i]):
                exposure = float(np.clip(
                    params.min_size + (params.max_size - params.min_size) * strength[i],
                    params.min_size, params.max_size,
                ))
                entry_bar = i + 1
                signal[i] = 1
                sizes[i] = exposure
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(season_window=15, atr_window=2)
    strategy = GeneratedStrategy()
    n = 26
    index = pd.bdate_range("2020-01-01", periods=n)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"atr": 2.0, "entry": False, "strength": 0.8}, index=index)
    decision = strategy.warmup_bars(params)
    ind.iloc[decision, ind.columns.get_loc("entry")] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=decision)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[decision:decision + 4].tolist() == [0, 1, 1, 0]
    assert np.isclose(result["size"].iloc[decision + 1], 0.85)
    # A gap changes the entry anchor: 110 - 2 = 108, so 107 must exit.
    data.iloc[decision + 1, data.columns.get_loc("open")] = 110.0
    data.iloc[decision + 1, data.columns.get_loc("close")] = 107.0
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].iloc[decision:decision + 4].tolist() == [0, 1, 0, 0]
    derived = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:22], params)
    pd.testing.assert_frame_equal(derived.iloc[:22], prefix)
    assert result["signal"].isin([0, 1]).all() and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
