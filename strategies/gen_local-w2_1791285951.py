from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    energy_window: int = 5
    rank_window: int = 60
    tail_percentile: float = 0.2
    rejection_threshold: float = 0.35
    vol_window: int = 20
    atr_window: int = 14
    target_daily_vol: float = 0.008
    min_size: float = 0.05
    max_size: float = 0.95
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791285951"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.energy_window + params.rank_window,
                       params.vol_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.energy_window, params.rank_window,
               params.vol_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (0 < params.tail_percentile < 0.5
                and 0 < params.rejection_threshold <= 1
                and 0 < params.min_size <= params.max_size <= 1
                and params.target_daily_vol > 0
                and params.stop_atr > 0
                and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid thresholds, sizing, stop, or holding horizon.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["up_energy"] = out["return"].clip(lower=0).pow(2).rolling(params.energy_window).mean()
        out["down_energy"] = out["return"].clip(upper=0).pow(2).rolling(params.energy_window).mean()
        out["total_energy"] = out["up_energy"] + out["down_energy"]
        out["balance"] = (out["up_energy"] - out["down_energy"]) / out["total_energy"].replace(0, np.nan)
        out["balance_rank"] = out["balance"].rolling(params.rank_window).rank(pct=True)
        out["range"] = data["high"] - data["low"]
        out["rejection"] = (data["close"] - data["open"]) / out["range"].where(out["range"] > 0)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["daily_vol"] = out["return"].rolling(params.vol_window).std(ddof=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        rank = indicators["balance_rank"]
        rejection = indicators["rejection"]
        candidate = np.select([
            (rank <= params.tail_percentile) & (rejection >= params.rejection_threshold),
            (rank >= 1 - params.tail_percentile) & (rejection <= -params.rejection_threshold)
        ], [1, -1], default=0)
        atr = indicators["atr"].to_numpy(dtype=float)
        vol = indicators["daily_vol"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        valid = (np.isfinite(rank.to_numpy(dtype=float))
                 & np.isfinite(rejection.to_numpy(dtype=float))
                 & np.isfinite(atr) & (atr > 0)
                 & np.isfinite(vol) & (vol > 0)
                 & np.isfinite(close) & (close > 0)
                 & np.isfinite(opening) & (opening > 0))
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        position = 0
        entry_bar = -1
        extreme = np.nan
        stop = np.nan
        trade_size = params.min_size
        for i in range(len(data)):
            if position:
                if not valid[i]:
                    position = 0
                    continue
                # Initialize at the actual next-bar opening fill, never the decision close.
                if i == entry_bar:
                    extreme = opening[i]
                if position == 1:
                    extreme = max(extreme, close[i])
                    proposed = extreme - params.stop_atr * atr[i]
                    stop = proposed if np.isnan(stop) else max(stop, proposed)
                    stopped = close[i] <= stop
                else:
                    # Mirror the high-water trail for shorts using the in-trade low.
                    extreme = min(extreme, close[i])
                    proposed = extreme + params.stop_atr * atr[i]
                    stop = proposed if np.isnan(stop) else min(stop, proposed)
                    stopped = close[i] >= stop
                if stopped or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
                    continue
            elif valid[i] and candidate[i]:
                position = int(candidate[i])
                entry_bar = i + 1
                extreme = np.nan
                stop = np.nan
                trade_size = float(np.clip(params.target_daily_vol / vol[i],
                                           params.min_size, params.max_size))
            signal[i] = position
            if position:
                size[i] = trade_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 103, 102, 100, 100, 97, 98, 100, 100], dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close,
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"balance_rank": [0.1, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5],
                        "rejection": [0.5, 0, 0, 0, -0.5, 0, 0, 0, 0],
                        "atr": [1, 1, 3, 1, 1, 1, 3, 1, 1],
                        "daily_vol": 0.02}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert np.allclose(result.loc[result["signal"] != 0, "size"], 0.4)
    ind["atr"] = 100.0
    ind["balance_rank"] = 0.1
    ind["rejection"] = 0.5
    timed = strategy.generate_signals(data, ind, None, params).data
    assert timed["signal"].tolist()[:6] == [0, 1, 1, 1, 1, 0]
    ind["rejection"] = -0.5
    assert not strategy.generate_signals(data, ind, None, params).data["signal"].any()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    empty = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert not empty["signal"].any()
    assert (empty["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
