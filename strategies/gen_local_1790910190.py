from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 5
    noise_bars: int = 20
    downside_bars: int = 10
    atr_bars: int = 14
    acceleration_threshold: float = 0.65
    downside_ratio: float = 0.85
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790910190"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, 2 * params.roc_bars + 1,
                       params.noise_bars + 1,
                       2 * params.downside_bars + 1,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.roc_bars, params.noise_bars, params.downside_bars,
               params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (np.isfinite(params.acceleration_threshold)
                and params.acceleration_threshold > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0
                and np.isfinite(params.downside_ratio)
                and 0 < params.downside_ratio <= 1):
            raise ValueError("Invalid acceleration, stop, or downside threshold.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"] - out["roc"].shift(params.roc_bars)
        out["noise"] = out["return"].rolling(params.noise_bars).std(ddof=0)
        scale = out["noise"] * np.sqrt(2.0 * params.roc_bars)
        out["acceleration_score"] = out["acceleration"] / scale.replace(0, np.nan)
        out["downside_energy"] = out["return"].clip(upper=0).pow(2)
        out["downside"] = out["downside_energy"].rolling(params.downside_bars).mean()
        out["prior_downside"] = out["downside"].shift(params.downside_bars)
        out["ma200"] = close.rolling(200).mean()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (close > out["ma200"])
            & (out["roc"] > 0)
            & (out["acceleration_score"] > params.acceleration_threshold)
            & (out["acceleration_score"].shift(1) <= params.acceleration_threshold)
            & (out["downside"] < params.downside_ratio * out["prior_downside"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            # A prior close's decision fills at this bar's open.
            if not active and i > 0 and signals[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                else:
                    continue
            if active:
                if closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": [100, 101, 105, 106, 104, 90, 92],
         "close": [100, 103, 110, 98, 96, 90, 92]}, index=index)
    indicators = pd.DataFrame(
        {"entry": [True, False, False, False, False, False, False],
         "atr": 2.0}, index=index)
    params = Params(stop_atr=2.0)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    params.hold_bars = 3
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
