from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    shock_window: int = 10
    rank_window: int = 126
    vol_window: int = 20
    atr_window: int = 14
    concentration_percentile: float = 0.90
    cooling_percentile: float = 0.50
    refractory_bars: int = 3
    holding_bars: int = 18
    stop_atr: float = 2.0
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790931642"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.shock_window + params.rank_window,
                       params.vol_window + 1, params.atr_window + 1,
                       params.refractory_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["energy"] = out["return"].pow(2)
        out["peak_energy"] = out["energy"].rolling(params.shock_window).max()
        out["total_energy"] = out["energy"].rolling(params.shock_window).sum()
        out["concentration"] = out["peak_energy"].div(out["total_energy"].replace(0, np.nan))
        out["concentration_rank"] = out["concentration"].rolling(params.rank_window).rank(pct=True)
        out["return_rank"] = out["return"].abs().rolling(params.rank_window).rank(pct=True)
        # Cubic returns identify the direction of the dominant variance contribution.
        out["dominance"] = out["return"].pow(3).rolling(params.shock_window).sum()
        bar = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        shock = out["energy"].eq(out["peak_energy"]) & out["energy"].gt(0)
        out["shock_bar"] = bar.where(shock).ffill()
        out["shock_age"] = bar - out["shock_bar"]
        out["repair"] = close.diff(params.refractory_bars) * np.sign(out["dominance"])
        out["volatility"] = out["return"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["size"] = (params.target_vol / out["volatility"].replace(0, np.nan)).clip(0.01, 1.0).fillna(1.0)
        eligible = (
            out["concentration_rank"].ge(params.concentration_percentile)
            & out["return_rank"].le(params.cooling_percentile)
            & out["shock_age"].ge(params.refractory_bars)
            & out["shock_age"].lt(params.shock_window)
            & out["repair"].lt(0)
            & out["atr"].gt(0)
            & out["volatility"].gt(0)
        )
        out["candidate"] = (-np.sign(out["dominance"])).where(eligible, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": indicators["size"].fillna(1.0)}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position, entry_bar, used_shock = 0, -1, -1.0
        stop, entry_atr = np.nan, np.nan
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            if position:
                if i == entry_bar:
                    # Observe the entry open only on its own bar; freeze decision-bar ATR.
                    entry = float(data["open"].iloc[i])
                    stop = entry - position * params.stop_atr * entry_atr
                stopped = (
                    not np.isfinite(close)
                    or not np.isfinite(stop)
                    or position * (close - stop) < 0
                )
                if stopped or i - entry_bar + 1 >= params.holding_bars:
                    position = 0
                    continue
            elif i >= self.warmup_bars(params) - 1 and np.isfinite(close):
                candidate = int(indicators["candidate"].iloc[i])
                shock = float(indicators["shock_bar"].iloc[i])
                if candidate and np.isfinite(shock) and shock > used_shock:
                    position = candidate
                    entry_bar = i + 1
                    entry_atr = float(indicators["atr"].iloc[i])
                    used_shock = shock
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(shock_window=2, rank_window=2, vol_window=2,
                    atr_window=2, holding_bars=3)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"candidate": 0, "shock_bar": 3.0,
                               "atr": 1.0, "size": 0.5}, index=index)
    indicators.loc[index[3], "candidate"] = 1
    indicators.loc[index[5], "candidate"] = 1
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    data.loc[index[4], ["open", "close"]] = [105.0, 102.0]
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0]
    indicators.loc[index[3], "candidate"] = -1
    data.loc[index[4], ["open", "close"]] = [95.0, 98.0]
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, -1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
