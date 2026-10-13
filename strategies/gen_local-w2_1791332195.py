from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_bars: int = 12
    breakout_bars: int = 20
    transmission_bars: int = 8
    atr_bars: int = 14
    innovation_min: float = 0.025
    transmission_min: float = 0.15
    stop_atr: float = 2.0
    hold_bars: int = 8
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791332195"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.trend_bars, params.breakout_bars,
                       params.transmission_bars, params.atr_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.trend_bars, params.breakout_bars,
               params.transmission_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if params.stop_atr <= 0 or params.hold_bars < 1 or params.size <= 0:
            raise ValueError("Stop, holding period and size must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["previous_close"] = close.shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["body"] = close - data["open"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars).mean()
        n = params.trend_bars
        x = np.arange(n, dtype=float) - (n - 1) / 2.0
        weights = x / np.dot(x, x)
        slope = np.full(len(data), np.nan)
        if len(data) >= n:
            slope[n - 1:] = np.convolve(close.to_numpy(), weights[::-1], mode="valid")
        out["latent_slope"] = slope
        out["latent_level"] = close.rolling(n, min_periods=n).mean() + out["latent_slope"] * (n - 1) / 2.0
        out["slope_innovation"] = out["latent_slope"].diff() / out["atr"].where(out["atr"] > 0)
        out["prior_channel"] = data["high"].rolling(
            params.breakout_bars, min_periods=params.breakout_bars).max().shift(1)
        m = params.transmission_bars
        out["transmission_numerator"] = (out["gap"] * out["body"]).rolling(m, min_periods=m).sum()
        out["transmission_denominator"] = (out["gap"].abs() * out["true_range"]).rolling(m, min_periods=m).sum()
        out["transmission"] = out["transmission_numerator"] / out["transmission_denominator"].where(out["transmission_denominator"] > 0)
        out["gap_sum"] = out["gap"].rolling(m, min_periods=m).sum()
        out["body_sum"] = out["body"].rolling(m, min_periods=m).sum()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["trend_breakout"] = (
            (close > out["prior_channel"])
            & (out["latent_slope"] > 0)
            & (out["slope_innovation"] > params.innovation_min)
        ).fillna(False)
        out["gap_transmission"] = (
            (out["transmission"] > params.transmission_min)
            & (out["gap_sum"] > 0)
            & (out["body_sum"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        setup = (indicators["trend_breakout"] & indicators["gap_transmission"]).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            exited = False
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Model the next-open fill; freeze the decision-bar ATR.
                    stop = opens[i] - params.stop_atr * entry_atr
                    entry_bar = i
                    active = True
                else:
                    exited = True
            if active:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                    exited = True
            if (not active and not exited and setup[i]
                    and np.isfinite(closes[i]) and closes[i] > 0
                    and np.isfinite(atrs[i]) and atrs[i] > 0):
                entry_atr = atrs[i]
                pending = True
            signal[i] = int(active or pending)
        df = pd.DataFrame({"signal": signal, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 105., 111., 108., 102., 100.],
        "close": [100., 110., 107., 101., 100., 100.],
    }, index=index)
    ind = pd.DataFrame({
        "trend_breakout": [True, False, False, False, False, False],
        "gap_transmission": [True, False, False, False, False, False],
        "atr": [2., 50., 50., 50., 50., 50.],
    }, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    ind["gap_transmission"] = False
    assert not strategy.generate_signals(data, ind, None, params).data["signal"].any()
    ind.iloc[0, ind.columns.get_loc("gap_transmission")] = True
    params.hold_bars = 2
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    t = np.arange(80, dtype=float)
    close = 100 + 0.15 * t + np.sin(t)
    bars = pd.DataFrame({
        "open": close - 0.2, "high": close + 1,
        "low": close - 1, "close": close, "volume": 1000000.,
    }, index=pd.date_range("2020-01-01", periods=80))
    full = strategy.indicators(bars, params)
    prefix = strategy.indicators(bars.iloc[:40], params)
    assert full.iloc[:40].equals(prefix)
    assert strategy.generate_signals(bars, full, None, params).data.iloc[:40].equals(
        strategy.generate_signals(bars.iloc[:40], prefix, None, params).data)
    assert not strategy.generate_signals(bars.iloc[:5], strategy.indicators(bars.iloc[:5], params), None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
