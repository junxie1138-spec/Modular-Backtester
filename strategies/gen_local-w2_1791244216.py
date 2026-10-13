from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 24
    range_window: int = 8
    baseline_window: int = 60
    atr_window: int = 14
    complexity_threshold: float = 0.055
    compression_ratio: float = 0.80
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791244216"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.pattern_window + 3, params.range_window + 1,
                       params.baseline_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.pattern_window, params.range_window,
               params.baseline_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not (0.0 < params.complexity_threshold < 1.0
                and 0.0 < params.compression_ratio < 1.0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0.0):
            raise ValueError("Invalid complexity, compression, or stop parameter")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        width = (data["high"] - data["low"]).astype(float)
        out["relative_range"] = (width / close.where(close > 0.0)).replace([np.inf, -np.inf], np.nan)
        out["short_range"] = out["relative_range"].rolling(params.range_window).mean()
        out["baseline_range"] = out["relative_range"].rolling(params.baseline_window).mean()
        out["compression"] = (out["short_range"] / out["baseline_range"].where(out["baseline_range"] > 0.0)).replace([np.inf, -np.inf], np.nan)

        triples = np.column_stack((out["relative_range"].shift(2),
                                   out["relative_range"].shift(1),
                                   out["relative_range"]))
        valid = np.isfinite(triples).all(axis=1)
        # Stable sorting resolves equal ranges chronologically.
        order = np.argsort(triples, axis=1, kind="stable")
        permutations = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                        (1, 2, 0), (2, 0, 1), (2, 1, 0))
        probability_columns = []
        for j, permutation in enumerate(permutations):
            column = f"pattern_probability_{j}"
            matches = (order == np.array(permutation)).all(axis=1).astype(float)
            matches[~valid] = np.nan
            out[column] = pd.Series(matches, index=data.index).rolling(params.pattern_window).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        entropy = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6)
        mixture = (probabilities + 1.0 / 6.0) / 2.0
        mixture_entropy = -(mixture * np.log(mixture)).sum(axis=1, min_count=6)
        out["ordinal_entropy"] = entropy / np.log(6.0)
        out["disequilibrium"] = ((mixture_entropy - 0.5 * entropy - 0.5 * np.log(6.0)) / np.log(2.0)).clip(lower=0.0)
        out["complexity"] = out["ordinal_entropy"] * out["disequilibrium"]
        out["setup_complexity"] = out["complexity"].shift(1)
        out["setup_compression"] = out["compression"].shift(1)
        out["channel_high"] = data["high"].rolling(params.range_window).max().shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat((width, (data["high"] - previous_close).abs(),
                                       (data["low"] - previous_close).abs()), axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry"] = ((out["setup_compression"] <= params.compression_ratio)
                        & (out["setup_complexity"] >= params.complexity_threshold)
                        & (close > out["channel_high"])
                        & (out["atr"] > 0.0)).fillna(False)
        out["conviction"] = (out["setup_complexity"] / (2.0 * params.complexity_threshold)).clip(lower=0.25, upper=1.0).fillna(0.25)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        count = len(data)
        signal = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        peak = stop = 0.0
        trade_size = 0.25
        for i in range(count):
            usable = np.isfinite(closes[i]) and np.isfinite(atr[i]) and closes[i] > 0.0 and atr[i] > 0.0
            if active:
                if np.isfinite(closes[i]) and closes[i] > 0.0:
                    peak = max(peak, closes[i])
                if usable:
                    stop = max(stop, peak - params.stop_atr * atr[i])
                stop_hit = np.isfinite(closes[i]) and closes[i] <= stop
                if stop_hit or i - entry_bar >= params.hold_bars:
                    active = False
            elif entries[i] and usable and np.isfinite(conviction[i]):
                active = True
                entry_bar = i
                peak = closes[i]
                stop = peak - params.stop_atr * atr[i]
                trade_size = float(np.clip(conviction[i], 0.25, 1.0))
            if active:
                signal[i] = 1
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": [100., 102., 103., 101., 100., 99., 100.],
                         "close": [100., 102., 103., 101., 100., 99., 100.],
                         "high": [101., 103., 104., 102., 101., 100., 101.],
                         "low": [99., 101., 102., 100., 99., 98., 99.],
                         "volume": 1000.}, index=index)
    indicators = pd.DataFrame({"atr": [1., 1., 1., 10., 10., 10., 10.],
                               "entry": [True, False, False, False, False, False, False],
                               "conviction": 0.7}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[index[1:5], "size"], 0.7)
    assert (result["size"] > 0.0).all()
    expiry = strategy.generate_signals(data, indicators, None, Params(hold_bars=2)).data
    assert expiry["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    quiet = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                          "close": 100., "volume": 1000.},
                         index=pd.date_range("2020-01-01", periods=100))
    derived = strategy.indicators(quiet, Params())
    assert not derived["entry"].any()
    assert derived["complexity"].dropna().eq(0.0).all()


if __name__ == "__main__":
    _self_check()
