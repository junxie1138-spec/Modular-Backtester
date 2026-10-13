from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRegimeParams:
    complexity_window: int = 12
    history_window: int = 126
    regime_quantile: float = 0.70
    tail_quantile: float = 0.20
    center_window: int = 5
    atr_window: int = 14
    atr_multiple: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalRegimeParams]):
    strategy_id = "gen_local_1790982553"

    @classmethod
    def params_type(cls) -> type[OrdinalRegimeParams]:
        return OrdinalRegimeParams

    def warmup_bars(self, params: OrdinalRegimeParams) -> int:
        return int(max(params.complexity_window + 3, params.center_window,
                       params.atr_window + 1) + params.history_window + 1)

    def indicators(self, data: pd.DataFrame, params: OrdinalRegimeParams) -> pd.DataFrame:
        if min(params.complexity_window, params.history_window,
               params.center_window, params.atr_window) < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not (0.5 < params.regime_quantile < 1.0 and
                0.0 < params.tail_quantile < 0.5 and
                np.isfinite(params.atr_multiple) and params.atr_multiple > 0.0 and
                3 <= params.holding_bars <= 5):
            raise ValueError("Invalid quantiles, ATR multiple, or holding horizon.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        triples = out[["return_lag2", "return_lag1", "return"]].to_numpy()
        order = np.argsort(triples, axis=1, kind="stable")
        codes = (9 * order[:, 0] + 3 * order[:, 1] + order[:, 2]).astype(float)
        codes[~np.isfinite(triples).all(axis=1)] = np.nan
        out["pattern"] = codes
        probabilities = []
        for code in (5, 7, 11, 15, 19, 21):
            occurrence = out["pattern"].eq(code).astype(float).where(out["pattern"].notna())
            name = f"pattern_probability_{code}"
            out[name] = occurrence.rolling(params.complexity_window,
                                           min_periods=params.complexity_window).mean()
            probabilities.append(name)
        probability = out[probabilities]
        terms = probability * np.log(probability.where(probability > 0.0, 1.0))
        out["complexity"] = -terms.sum(axis=1, min_count=6) / np.log(6.0)

        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window,
                                               min_periods=params.atr_window).mean()
        out["center"] = close.rolling(params.center_window,
                                     min_periods=params.center_window).mean()
        out["strain"] = ((close - out["center"]) / out["atr"].where(out["atr"] > 0.0)).replace(
            [np.inf, -np.inf], np.nan)
        complexity_history = out["complexity"].shift(1).rolling(
            params.history_window, min_periods=params.history_window)
        strain_history = out["strain"].shift(1).rolling(
            params.history_window, min_periods=params.history_window)
        out["complexity_low"] = complexity_history.quantile(1.0 - params.regime_quantile)
        out["complexity_high"] = complexity_history.quantile(params.regime_quantile)
        out["strain_low"] = strain_history.quantile(params.tail_quantile)
        out["strain_high"] = strain_history.quantile(1.0 - params.tail_quantile)

        reversal = ((out["complexity"] > out["complexity_high"]) &
                    (out["strain"] < out["strain_low"]) & (out["strain"] < 0.0))
        continuation = ((out["complexity"] < out["complexity_low"]) &
                        (out["strain"] > out["strain_high"]) & (out["strain"] > 0.0))
        ready = np.isfinite(out[["return", "atr", "strain", "complexity",
                                "complexity_low", "complexity_high",
                                "strain_low", "strain_high"]]).all(axis=1)
        out["entry"] = (ready & (out["atr"] > 0.0) & (out["return"] > 0.0) &
                        (reversal | continuation)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalRegimeParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0.0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.atr_multiple * atrs[i])
                    if closes[i] <= stop or age >= params.holding_bars:
                        active = False
            elif valid and entries[i]:
                active = True
                age = 0
                # The entry fills next bar; its first close starts the in-trade high.
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    dates = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"close": [100.0, 101.0, 103.0, 102.0, 101.0, 100.0]}, index=dates)
    features = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
                             "entry": [True, False, False, False, False, False]}, index=dates)
    result = strategy.generate_signals(prices, features, None,
                                      OrdinalRegimeParams(holding_bars=5))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    prices["close"] = 100.0
    features["atr"] = 1.0
    result = strategy.generate_signals(prices, features, None, OrdinalRegimeParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
