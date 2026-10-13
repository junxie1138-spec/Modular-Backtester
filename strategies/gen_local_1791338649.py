from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    variance_bars: int = 8
    rank_bars: int = 63
    downside_percentile: float = 0.20
    pressure_bars: int = 3
    pressure_threshold: float = 0.20
    vol_bars: int = 20
    target_vol: float = 0.10
    atr_bars: int = 14
    breakeven_gain: float = 0.0075
    trail_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791338649"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.variance_bars + params.rank_bars,
                       params.vol_bars + 1, params.atr_bars + 1,
                       params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change(fill_method=None)
        squared = returns.pow(2)
        total = squared.rolling(params.variance_bars).sum()
        downside = returns.clip(upper=0).pow(2).rolling(params.variance_bars).sum()
        downside_share = downside.div(total.replace(0, np.nan))
        candle_range = (data["high"] - data["low"]).replace(0, np.nan)
        placement = ((2 * close - data["high"] - data["low"]) / candle_range).clip(-1, 1)
        previous = close.shift(1)
        true_range = pd.concat([(data["high"] - data["low"]),
                                (data["high"] - previous).abs(),
                                (data["low"] - previous).abs()], axis=1).max(axis=1)
        return pd.DataFrame({
            "returns": returns,
            "squared_return": squared,
            "total_variance": total,
            "downside_variance": downside,
            "downside_share": downside_share,
            "downside_rank": downside_share.rolling(params.rank_bars).rank(pct=True),
            "close_placement": placement,
            "pressure": placement.rolling(params.pressure_bars).mean(),
            "true_range": true_range,
            "atr": true_range.rolling(params.atr_bars).mean(),
            "annual_vol": returns.rolling(params.vol_bars).std(ddof=0) * np.sqrt(252),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = ((indicators["downside_rank"] <= params.downside_percentile)
                   & (indicators["pressure"] >= params.pressure_threshold)
                   & (indicators["atr"] > 0)
                   & (indicators["annual_vol"] > 0)).fillna(False).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        volatility = indicators["annual_vol"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        active = pending = armed = False
        entry = stop = peak = 0.0
        age = 0
        weight = 0.95
        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    age = 0
                    armed = False
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]):
                    active = False
                else:
                    peak = max(peak, closes[i])
                    if closes[i] >= entry * (1 + params.breakeven_gain):
                        armed = True
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    # Stops are evaluated at the close; exits fill next bar.
                    if closes[i] <= stop or age >= params.hold_bars:
                        active = False
                if active:
                    signal[i] = 1
                    sizes[i] = weight
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                weight = float(np.clip(params.target_vol / volatility[i], 0.05, 0.95))
                pending = True
                signal[i] = 1
                sizes[i] = weight
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(trail_atr=1.0, breakeven_gain=0.01)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0,
                         "close": [100., 100., 102., 101., 100., 100., 100., 100.],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"downside_rank": [0.1] + [0.9] * 7,
                               "pressure": 0.4, "atr": 1.0, "annual_vol": 0.2},
                              index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert (result.loc[result["signal"] == 1, "size"] == 0.5).all()
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    prefix = strategy.generate_signals(data.iloc[:3], indicators.iloc[:3], None, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:3])


if __name__ == "__main__":
    _self_check()
