from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    regression_window: int = 40
    entry_score: float = 0.30
    target_vol: float = 0.12
    breakeven_pct: float = 0.025
    atr_mult: float = 3.0
    max_hold: int = 18

    def __post_init__(self):
        if self.regression_window < 10 or self.max_hold < 1:
            raise ValueError("Invalid lookback or holding period")
        if not all(np.isfinite(x) and x > 0 for x in (
            self.entry_score, self.target_vol, self.breakeven_pct, self.atr_mult
        )):
            raise ValueError("Thresholds and risk parameters must be finite and positive")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790903655"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(2 * params.regression_window + 30)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        out["latent_slope"] = out["return"].ewm(
            span=8, adjust=False, min_periods=8
        ).mean()
        out["slope_innovation"] = out["latent_slope"].diff()
        out["variance_baseline"] = out["return"].pow(2).ewm(
            span=20, adjust=False, min_periods=20
        ).mean()
        out["variance_innovation"] = (
            out["return"].pow(2) - out["variance_baseline"].shift(1)
        )
        window = params.regression_window
        out["innovation_cov"] = out["slope_innovation"].rolling(window).cov(
            out["variance_innovation"]
        )
        out["variance_noise"] = out["variance_innovation"].rolling(window).var()
        out["beta"] = (
            out["innovation_cov"] / out["variance_noise"].clip(lower=1e-16)
        ).shift(1)
        out["residual"] = (
            out["slope_innovation"] - out["beta"] * out["variance_innovation"]
        )
        out["residual_mean"] = out["residual"].ewm(
            span=4, adjust=False, min_periods=4
        ).mean()
        out["residual_noise"] = out["residual"].rolling(window).std()
        out["score"] = (
            out["residual_mean"] / out["residual_noise"].clip(lower=1e-10)
        )
        out["annual_vol"] = out["return"].rolling(20).std() * np.sqrt(252.0)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        score = indicators["score"].fillna(0.0)
        strength = score.clip(lower=0.05)
        volatility = indicators["annual_vol"].clip(lower=0.01)
        df["size"] = (
            params.target_vol / volatility * strength / (1.0 + strength)
        ).clip(lower=0.01, upper=1.0).fillna(0.01)
        ready = np.isfinite(indicators[
            ["score", "latent_slope", "annual_vol", "atr"]
        ]).all(axis=1)
        enter = (
            ready & (score >= params.entry_score)
            & (indicators["latent_slope"] > 0)
            & (indicators["atr"] > 0)
        ).to_numpy()
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = stop = 0.0
        age = 0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            o, h, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and o > 0 and close > 0
            if pending:
                pending = False
                if valid and i > 0 and np.isfinite(atr[i - 1]):
                    entry = o
                    stop = entry - params.atr_mult * atr[i - 1]
                    active = True
                    armed = False
                    age = 0
            if active:
                age += 1
                if not valid or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                elif low <= stop or age >= params.max_hold:
                    active = False
                else:
                    # Raise only after evaluating the stop carried into this bar.
                    if close >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, close - params.atr_mult * atr[i])
                    signals[i] = 1
                # An exit cannot immediately reenter on the same decision bar.
                continue
            if i >= warmup - 1 and valid and enter[i]:
                signals[i] = 1
                pending = True
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    w = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=w + 25, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "score": 0.0, "latent_slope": 0.001,
        "annual_vol": 0.2, "atr": 1.0,
    }, index=index)
    indicators.iloc[w, indicators.columns.get_loc("score")] = 1.0
    ctx = StrategyContext("SPY", "1d", w)
    baseline = strategy.generate_signals(data, indicators, ctx, params).data
    assert baseline["signal"].iloc[w] == 0
    assert baseline["signal"].iloc[w + 1] == 1
    assert baseline["signal"].iloc[w + 18] == 1
    assert baseline["signal"].iloc[w + 19] == 0
    for offset, price in [(2, 103.0), (3, 108.0), (4, 107.0), (5, 104.0)]:
        data.iloc[w + offset, :4] = [price, price + 1, price - 1, price]
    indicators.iloc[w + 4, indicators.columns.get_loc("atr")] = 4.0
    stopped = strategy.generate_signals(data, indicators, ctx, params).data
    assert stopped["signal"].iloc[w + 5] == 1
    assert stopped["signal"].iloc[w + 6] == 0
    assert stopped["signal"].isin([0, 1]).all()
    assert np.isfinite(stopped["size"]).all() and (stopped["size"] > 0).all()
    assert strategy.indicators(data, params).index.equals(index)


if __name__ == "__main__":
    _self_check()
