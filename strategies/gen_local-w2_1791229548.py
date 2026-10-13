from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 12
    breakout_bars: int = 20
    area_threshold: float = 0.08
    atr_bars: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791229548"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(200, params.path_bars + 1, params.breakout_bars + 1,
                   params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        log_price = np.log(close)
        n = params.path_bars
        start = log_price.shift(n)
        # Antisymmetric second-order signature of (normalized time, log price).
        trapezoid_mean = (log_price.rolling(n, min_periods=n).sum()
                          - 0.5 * log_price + 0.5 * start) / n
        area = 0.5 * (log_price + start) - trapezoid_mean
        variation = log_price.diff().abs().rolling(n, min_periods=n).sum()
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        result = pd.DataFrame(index=data.index)
        result["log_price"] = log_price
        result["path_area"] = area
        result["variation"] = variation
        result["area_score"] = area / variation.where(variation > 0)
        result["true_range"] = true_range
        result["atr"] = true_range.rolling(params.atr_bars,
                                           min_periods=params.atr_bars).mean()
        result["ma200"] = close.rolling(200, min_periods=200).mean()
        result["ceiling"] = data["high"].shift(1).rolling(
            params.breakout_bars, min_periods=params.breakout_bars).max()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        score = indicators["area_score"].to_numpy(dtype=float)
        ceiling = indicators["ceiling"].to_numpy(dtype=float)
        ready = (np.isfinite(close) & np.isfinite(atr) & (atr > 0)
                 & np.isfinite(regime) & np.isfinite(score)
                 & np.isfinite(ceiling))
        signal = np.zeros(len(data), dtype=int)
        active = False
        armed = True
        high_water = 0.0
        stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if not ready[i]:
                active = False
                continue
            if active:
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if (close[i] <= stop or close[i] <= regime[i]
                        or i - entry_bar >= params.hold_bars):
                    active = False
                    continue
                signal[i] = 1
            else:
                # Rearm only after the time-price area loses its positive bias.
                if score[i] <= 0:
                    armed = True
                if (armed and close[i] > regime[i] and close[i] > ceiling[i]
                        and score[i] >= params.area_threshold):
                    active = True
                    armed = False
                    high_water = close[i]
                    stop = high_water - params.trail_atr * atr[i]
                    entry_bar = i
                    signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = SignatureParams()
    s = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=230, freq="B")
    close = np.full(230, 100.0)
    close[200:204] = [101.0, 103.0, 102.0, 99.0]
    data = pd.DataFrame({"open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close,
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "ma200": 90.0,
                        "area_score": 0.1, "ceiling": 100.0}, index=index)
    ind.loc[index[202:204], "atr"] = 10.0
    result = s.generate_signals(data, ind, None, p).data
    assert result["signal"].iloc[200:205].tolist() == [0, 1, 1, 1, 0]
    ind["atr"] = 1.0
    data.loc[index[200:], "close"] = 101.0
    result = s.generate_signals(data, ind, None, p).data
    assert result["signal"].iloc[201:221].eq(1).all()
    assert result["signal"].iloc[221] == 0
    assert s.indicators(data, p).index.equals(index)
    assert s.warmup_bars(p) >= 200
