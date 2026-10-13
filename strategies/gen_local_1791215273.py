from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 6
    percentile_window: int = 126
    entry_percentile: float = 0.8
    volatility_window: int = 20
    atr_window: int = 14
    annual_vol_target: float = 0.12
    trailing_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791215273"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(max(params.path_window + params.percentile_window + 1,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        if (params.path_window < 2 or params.percentile_window < 2
                or params.volatility_window < 2 or params.atr_window < 1
                or not 0 < params.entry_percentile < 1
                or not 3 <= params.hold_bars <= 5
                or not np.isfinite(params.annual_vol_target)
                or params.annual_vol_target <= 0
                or not np.isfinite(params.trailing_atr)
                or params.trailing_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["variation"] = out["return"].abs()
        # Exact second-order antisymmetric signature of (return, absolute return).
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.path_window):
            cross = (out["return"].shift(lag) * out["variation"]
                     - out["variation"].shift(lag) * out["return"])
            area = area + 0.5 * cross.rolling(params.path_window - lag).sum()
        out["total_variation"] = out["variation"].rolling(params.path_window).sum()
        out["area"] = area / out["total_variation"].pow(2).replace(0, np.nan)
        out["area_threshold"] = out["area"].abs().shift(1).rolling(
            params.percentile_window).quantile(params.entry_percentile)
        out["annual_vol"] = out["return"].rolling(params.volatility_window).std() * np.sqrt(252.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["target_size"] = (params.annual_vol_target / out["annual_vol"].replace(0, np.nan)).clip(0.05, 0.95)
        ready = (out[["area", "area_threshold", "annual_vol", "atr", "target_size"]]
                 .replace([np.inf, -np.inf], np.nan).notna().all(axis=1))
        eligible = ready & (out["atr"] > 0) & (out["annual_vol"] > 0) & (out["area"].abs() > out["area_threshold"])
        out["entry"] = np.select([
            eligible & (out["area"] < 0) & (out["return"] > 0),
            eligible & (out["area"] > 0) & (out["return"] < 0),
        ], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        high_water, stop = np.nan, -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if side:
                # This is the first held bar after yesterday's entry decision.
                age += 1
                if not valid:
                    side = 0
                else:
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    stop = max(stop, high_water - params.trailing_atr * atr)
                    # Apply the requested highest-close drawdown exit to BOTH directions.
                    if close <= stop or age >= params.hold_bars:
                        side = 0
            elif valid and entries[i]:
                side = int(entries[i])
                age = 0
                high_water, stop = np.nan, -np.inf
            signals[i] = side
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["target_size"].shift(1).replace(
            [np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 100., 110., 109., 107., 107., 107.]}, index=index)
    ind = pd.DataFrame({"atr": 1., "entry": [1, 0, 0, 0, 0, 0, 0],
                        "target_size": 0.4}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    # ATR expansion cannot loosen the stop set at the highest close.
    ind["atr"] = [1., 1., 1., 10., 10., 10., 10.]
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    ind["entry"] = [-1, 0, 0, 0, 0, 0, 0]
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, -1, -1, -1, -1, 0, 0]
    ind["entry"] = [1, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    short = pd.DataFrame({"open": [100., 101.], "high": [101., 102.],
                          "low": [99., 100.], "close": [100., 101.],
                          "volume": [1000., 1000.]}, index=index[:2])
    warm = strategy.indicators(short, params)
    assert warm["entry"].eq(0).all()
    assert strategy.generate_signals(short, warm, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
