from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    path_window: int = 6
    rank_window: int = 126
    area_quantile: float = 0.8
    corr_window: int = 20
    atr_period: int = 14
    breakeven_pct: float = 0.0075
    trail_atr: float = 1.7
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790942403"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(
            params.path_window + params.rank_window + 2,
            params.corr_window + 2,
            params.atr_period + 1,
        )

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.path_window, params.rank_window, params.corr_window, params.atr_period) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not 0.0 < params.area_quantile < 1.0:
            raise ValueError("area_quantile must be between zero and one.")
        if params.trail_atr <= 0 or params.breakeven_pct <= 0 or params.hold_bars < 1:
            raise ValueError("Exit parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["lag_return"] = out["return"].shift(1)
        x, y = out["lag_return"], out["return"]
        # Antisymmetric second-order signature of cumulative (lagged, current) returns.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.path_window):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + 0.5 * cross.rolling(params.path_window - lag).sum()
        out["signature_area"] = area
        out["path_energy"] = (x.pow(2) + y.pow(2)).rolling(params.path_window).sum()
        out["clockwise_area"] = -area / out["path_energy"].where(out["path_energy"] > 0)
        out["area_threshold"] = out["clockwise_area"].shift(1).rolling(params.rank_window).quantile(params.area_quantile)
        out["autocorrelation"] = y.rolling(params.corr_window).corr(x)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        ready = np.isfinite(indicators[
            ["clockwise_area", "area_threshold", "autocorrelation", "return", "lag_return", "atr"]
        ]).all(axis=1)
        enter = (
            ready & (indicators["clockwise_area"] > indicators["area_threshold"])
            & (indicators["autocorrelation"] < 0)
            & (indicators["return"] > 0) & (indicators["lag_return"] <= 0)
            & (indicators["atr"] > 0)
        ).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = peak = stop = entry_atr = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * entry_atr
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                peak = max(peak, closes[i])
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # Stops are evaluated at the close; the zero target executes next bar.
                if closes[i] <= stop or age >= params.hold_bars:
                    active = False
                    continue
                signal[i] = 1
            elif i >= self.warmup_bars(params) and enter[i]:
                signal[i] = 1
                entry_atr = atr[i]
                pending = True
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(path_window=2, rank_window=3, corr_window=3, atr_period=2)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=20)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 98.0, "close": 100.1, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame(
        {"clockwise_area": 0.0, "area_threshold": 1.0, "autocorrelation": -0.5,
         "return": 0.01, "lag_return": -0.01, "atr": 0.5}, index=index,
    )
    ind.loc[index[7], "clockwise_area"] = 2.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[:8].eq(0).all()
    assert result["signal"].iloc[8:12].eq(1).all() and result["signal"].iloc[12] == 0
    data.loc[index[8:11], "close"] = [102.0, 101.3, 101.0]
    ind.loc[index[9:11], "atr"] = 5.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[8:11].eq(1).all() and result["signal"].iloc[11] == 0
    data.loc[index[8:10], "close"] = [100.8, 99.9]
    ind.loc[index[8:10], "atr"] = 2.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[9] == 1 and result["signal"].iloc[10] == 0
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
