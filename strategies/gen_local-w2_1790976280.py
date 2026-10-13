from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    complexity_window: int = 36
    complexity_threshold: float = 0.90
    saturation_bars: int = 3
    breakout_window: int = 20
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790976280"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.complexity_window + params.saturation_bars + 1,
                       params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.complexity_window, params.saturation_bars,
               params.breakout_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not 0.0 <= params.complexity_threshold <= 1.0:
            raise ValueError("Complexity threshold must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("ATR stop multiplier must be finite and positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        out["pattern"] = np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan)
        valid = out["pattern"].notna()
        out["pattern_count"] = valid.astype(float).rolling(
            params.complexity_window, min_periods=params.complexity_window).sum()
        entropy = pd.Series(0.0, index=data.index)
        for motif in range(6):
            probability = out["pattern"].eq(motif).astype(float).rolling(
                params.complexity_window, min_periods=params.complexity_window).sum()
            probability = probability.div(out["pattern_count"].replace(0.0, np.nan))
            out[f"motif_probability_{motif}"] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out["complexity"] = (entropy / np.log(6.0)).where(
            out["pattern_count"].eq(params.complexity_window))
        out["saturated"] = out["complexity"].rolling(
            params.saturation_bars, min_periods=params.saturation_bars).min().ge(
                params.complexity_threshold)
        out["upper"] = data["high"].rolling(
            params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        out["lower"] = data["low"].rolling(
            params.breakout_window, min_periods=params.breakout_window).min().shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["true_range"] = out["true_range"].where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        ready = np.isfinite(out["atr"]) & out["atr"].gt(0.0)
        out["long_entry"] = (ready & out["saturated"] & close.gt(out["upper"])
                             & close.gt(out["ma200"]))
        out["short_entry"] = (ready & out["saturated"] & close.lt(out["lower"])
                              & close.lt(out["ma200"]))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        age = 0

        for i in range(len(data)):
            exited = False
            if position:
                # A decision from the previous close fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        exited = True
                if position:
                    age += 1
                    stop = entry_price - position * params.stop_atr * entry_atr
                    stopped = np.isfinite(closes[i]) and (
                        (position == 1 and closes[i] < stop)
                        or (position == -1 and closes[i] > stop))
                    if stopped or age >= params.hold_bars:
                        position = 0
                        exited = True
            if position == 0 and not exited:
                if np.isfinite(atr[i]) and atr[i] > 0.0:
                    if longs[i]:
                        position = 1
                    elif shorts[i]:
                        position = -1
                    if position:
                        entry_price = np.nan
                        entry_atr = atr[i]
                        age = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=33, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 110.0
    data.loc[index[3], "close"] = 99.0
    data.loc[index[4], "close"] = 97.0
    data.loc[index[8], "close"] = 103.0
    indicators = pd.DataFrame({"atr": 1.0, "long_entry": False,
                               "short_entry": False}, index=index)
    indicators.loc[index[[0, 10]], "long_entry"] = True
    indicators.loc[index[6], "short_entry"] = True
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, StrategyParams()).data
    signal = result["signal"]
    assert signal.iloc[0] == 0 and signal.iloc[1] == 1
    assert signal.iloc[3] == 1 and signal.iloc[4] == 1 and signal.iloc[5] == 0
    assert signal.iloc[7] == -1 and signal.iloc[8] == -1 and signal.iloc[9] == 0
    assert signal.iloc[11:29].eq(1).all() and signal.iloc[29] == 0
    assert signal.isin([-1, 0, 1]).all() and result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
