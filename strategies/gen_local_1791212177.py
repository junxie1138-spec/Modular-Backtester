from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 40
    percentile_window: int = 126
    breakout_percentile: float = 0.85
    participation_window: int = 10
    participation_lag: int = 5
    minimum_participation: float = 0.6
    atr_window: int = 14
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791212177"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(
            params.ma_window + params.percentile_window + 1,
            params.ma_window + params.participation_window + params.participation_lag,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.ma_window, params.percentile_window,
               params.participation_window, params.participation_lag,
               params.atr_window, params.holding_bars) < 1:
            raise ValueError("Window lengths and holding_bars must be positive")
        if params.ma_window < 2:
            raise ValueError("ma_window must be at least two")
        if not 0.0 < params.breakout_percentile < 1.0:
            raise ValueError("breakout_percentile must be between zero and one")
        if not 0.0 <= params.minimum_participation <= 1.0:
            raise ValueError("minimum_participation must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive and finite")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["threshold"] = out["z"].rolling(params.percentile_window).quantile(params.breakout_percentile).shift(1)
        out["above_ma"] = out["z"].gt(0.0).astype(float).where(out["z"].notna())
        out["participation"] = out["above_ma"].rolling(params.participation_window).mean()
        out["participation_gain"] = out["participation"] - out["participation"].shift(params.participation_lag)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry"] = (
            out["z"].gt(0.0)
            & out["z"].gt(out["threshold"])
            & out["z"].shift(1).le(out["threshold"].shift(1))
            & out["participation"].ge(params.minimum_participation)
            & out["participation_gain"].gt(0.0)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    active = False
                elif (
                    (np.isfinite(closes[i]) and closes[i] < entry_price - params.stop_atr * entry_atr)
                    or i - entry_bar + 1 >= params.holding_bars
                ):
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                entry_atr = atr[i]
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0],
        "high": [102.0] * 6,
        "low": [97.0] * 6,
        "close": [100.0, 100.0, 99.0, 98.0, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": [1.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, GeneratedParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    result = GeneratedStrategy.generate_signals(data, indicators, None, GeneratedParams(holding_bars=2))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    computed = GeneratedStrategy.indicators(flat, GeneratedParams())
    assert not computed["entry"].any()


if __name__ == "__main__":
    _self_check()
