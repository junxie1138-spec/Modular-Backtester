from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 6
    volume_window: int = 20
    atr_window: int = 14
    efficiency_min: float = 0.45
    concordance_min: float = 0.20
    volume_multiple: float = 1.10
    trail_atr: float = 1.60
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790984741"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.volume_window + params.trend_window,
                       params.atr_window + 1, params.trend_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if min(p.trend_window, p.volume_window, p.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (3 <= p.max_hold <= 5 and p.trail_atr > 0
                and 0 < p.efficiency_min <= 1
                and 0 < p.concordance_min <= 1 and p.volume_multiple > 0):
            raise ValueError("Invalid holding period or signal thresholds.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["delta"] = data["close"].diff()
        out["return"] = data["close"].div(out["previous_close"].replace(0, np.nan)) - 1
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        out["volume_baseline"] = data["volume"].shift(1).rolling(p.volume_window).mean()
        out["relative_volume"] = data["volume"].div(out["volume_baseline"].replace(0, np.nan))
        out["displacement"] = data["close"].diff(p.trend_window)
        out["path_length"] = out["delta"].abs().rolling(p.trend_window).sum()
        out["efficiency"] = out["displacement"].div(out["path_length"].replace(0, np.nan))
        out["concordance"] = out["return"].rolling(p.trend_window).corr(out["relative_volume"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        p = params
        columns = ["ma200", "atr", "delta", "efficiency", "concordance", "relative_volume"]
        ready = indicators[columns].notna().all(axis=1) & (indicators["atr"] > 0)
        confirmed = ready & (indicators["relative_volume"] >= p.volume_multiple)
        long_entry = (confirmed & (data["close"] > indicators["ma200"])
                      & (indicators["delta"] > 0)
                      & (indicators["efficiency"] >= p.efficiency_min)
                      & (indicators["concordance"] >= p.concordance_min))
        short_entry = (confirmed & (data["close"] < indicators["ma200"])
                       & (indicators["delta"] < 0)
                       & (indicators["efficiency"] <= -p.efficiency_min)
                       & (indicators["concordance"] <= -p.concordance_min))
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        valid = ready.to_numpy() & np.isfinite(close) & (close > 0)
        longs = long_entry.fillna(False).to_numpy()
        shorts = short_entry.fillna(False).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position = age = 0
        extreme = stop = np.nan
        for i in range(len(data)):
            if not valid[i]:
                position = age = 0
                extreme = stop = np.nan
                continue
            if position:
                # The first held close follows the entry decision by one bar.
                age += 1
                if position == 1:
                    extreme = close[i] if age == 1 else max(extreme, close[i])
                    candidate = extreme - p.trail_atr * atr[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close[i] <= stop or close[i] <= regime[i]
                else:
                    # Mirror the long high-water stop for short positions.
                    extreme = close[i] if age == 1 else min(extreme, close[i])
                    candidate = extreme + p.trail_atr * atr[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close[i] >= stop or close[i] >= regime[i]
                if stopped or age >= p.max_hold:
                    position = age = 0
                    extreme = stop = np.nan
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
                extreme = stop = np.nan
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"close": [100, 104, 105, 103, 102, 100, 96, 99, 99]}, index=index)
    indicators = pd.DataFrame({
        "ma200": [90] * 5 + [110] * 4,
        "atr": [1, 1, 1, 10, 10, 1, 1, 10, 10],
        "delta": [1, 4, 1, -2, -1, -2, -4, 3, 0],
        "efficiency": [1] * 5 + [-1] * 4,
        "concordance": [1] * 5 + [-1] * 4,
        "relative_volume": 2.0,
    }, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, 0]
    indicators.loc[:, ["ma200", "atr", "delta", "efficiency", "concordance"]] = [90, 1, 1, 1, 1]
    data["close"] = 100
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist()[:6] == [0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
