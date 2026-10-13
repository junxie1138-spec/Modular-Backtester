from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    graph_bars: int = 12
    percentile_bars: int = 126
    recovery_bars: int = 20
    percentile: float = 0.85
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790862888"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.graph_bars, params.recovery_bars) + params.percentile_bars + 1)

    @staticmethod
    def _bridge_score(values: pd.Series, window: int) -> pd.Series:
        blocker = pd.Series(-np.inf, index=values.index)
        degree = pd.Series(0.0, index=values.index)
        total_length = pd.Series(0.0, index=values.index)
        longest = pd.Series(0.0, index=values.index)
        for lag in range(1, window):
            other = values.shift(lag)
            visible = values.notna() & other.notna() & (values > blocker) & (other > blocker)
            degree += visible.astype(float)
            total_length += lag * visible.astype(float)
            longest = longest.where(~visible, float(lag))
            blocker = pd.concat([blocker, other], axis=1).max(axis=1)
        score = longest - total_length / degree.replace(0.0, np.nan)
        return score.where(values.rolling(window, min_periods=window).count() == window)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.graph_bars < 3 or params.recovery_bars < 2 or params.percentile_bars < 2:
            raise ValueError("Lookbacks must support complete topology and percentile windows.")
        if not 0.0 < params.percentile < 1.0 or not 3 <= params.holding_bars <= 5:
            raise ValueError("Require a percentile in (0, 1) and a holding period of 3-5 bars.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["long_bridge"] = self._bridge_score(data["low"], params.graph_bars)
        out["short_bridge"] = self._bridge_score(-data["high"], params.graph_bars)
        peak = close.rolling(params.recovery_bars, min_periods=params.recovery_bars).max()
        trough = close.rolling(params.recovery_bars, min_periods=params.recovery_bars).min()
        out["peak"] = peak
        out["trough"] = trough
        out["drawdown"] = 1.0 - close / peak
        out["runup"] = close / trough - 1.0
        out["change"] = close.diff()
        for name in ("long_bridge", "short_bridge", "drawdown", "runup"):
            out[name + "_threshold"] = out[name].shift(1).rolling(
                params.percentile_bars, min_periods=params.percentile_bars
            ).quantile(params.percentile)
        out["long_setup"] = (
            (out["drawdown"].shift(1) > out["drawdown_threshold"].shift(1))
            & (out["drawdown"] > 0.0)
            & (out["change"] > 0.0)
            & (out["long_bridge"] > out["long_bridge_threshold"])
        ).fillna(False)
        out["short_setup"] = (
            (out["runup"].shift(1) > out["runup_threshold"].shift(1))
            & (out["runup"] > 0.0)
            & (out["change"] < 0.0)
            & (out["short_bridge"] > out["short_bridge_threshold"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        long_setup = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        short_setup = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar in range(len(data)):
            if position and bar == exit_bar:
                position = 0
            elif not position:
                if long_setup[bar] and not short_setup[bar]:
                    position = 1
                elif short_setup[bar] and not long_setup[bar]:
                    position = -1
                if position:
                    exit_bar = bar + params.holding_bars
            signal[bar] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    values = pd.Series([10.0, 2.0, 3.0, 8.0])
    assert np.isclose(strategy._bridge_score(values, 4).iloc[-1], 1.0)
    params = StrategyParams()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=15))
    setups = pd.DataFrame({"long_setup": False, "short_setup": False}, index=data.index)
    setups.iloc[0, 0] = True
    setups.iloc[2, 1] = True
    setups.iloc[4, 0] = True
    setups.iloc[6, 1] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    actual = strategy.generate_signals(data, setups, ctx, params).data
    assert actual["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0]
    assert (actual["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
