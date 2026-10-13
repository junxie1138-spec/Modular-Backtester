from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 63
    rank_window: int = 126
    recovery_bars: int = 3
    extreme_percentile: float = 0.85
    rank_release: float = 0.12
    min_drawdown: float = 0.02
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790924015"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.peak_window + params.rank_window + params.recovery_bars)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.peak_window, params.rank_window, params.recovery_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0 < params.extreme_percentile < 1 and 0 < params.rank_release <= 1 and 0 <= params.min_drawdown < 1):
            raise ValueError("Invalid percentile or drawdown threshold.")
        close = data["close"]
        peak = close.rolling(params.peak_window).max()
        depth = (1.0 - close / peak.replace(0.0, np.nan)).clip(lower=0.0)
        rank = depth.rolling(params.rank_window).rank(pct=True)
        prior_rank = rank.shift(params.recovery_bars)
        release = prior_rank - rank
        repair = depth.shift(params.recovery_bars) - depth
        stable_peak = peak.eq(peak.shift(params.recovery_bars))
        close_up = close.gt(close.shift(1))
        entry = (
            prior_rank.ge(params.extreme_percentile)
            & release.ge(params.rank_release)
            & repair.gt(0.0)
            & depth.ge(params.min_drawdown)
            & stable_peak
            & close_up
        )
        strength = (release / 0.40).clip(0.0, 1.0)
        return pd.DataFrame(
            {
                "peak": peak,
                "drawdown": depth,
                "drawdown_rank": rank,
                "prior_rank": prior_rank,
                "rank_release": release,
                "depth_repair": repair,
                "stable_peak": stable_peak,
                "close_up": close_up,
                "entry": entry,
                "strength": strength,
            },
            index=data.index,
        ).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        exit_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
            elif entry[i]:
                exit_bar = i + params.hold_bars
                entry_size = 0.25 + 0.75 * strength[i]
            if exit_bar >= 0:
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=params.hold_bars + 8, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": False, "strength": 0.0}, index=index)
    indicators.iloc[2] = [True, 0.5]
    indicators.iloc[2 + params.hold_bars] = [True, 1.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:3].eq(0).all()
    assert result["signal"].iloc[3:3 + params.hold_bars].eq(1).all()
    assert result["signal"].iloc[3 + params.hold_bars:].eq(0).all()
    assert result["size"].iloc[3:3 + params.hold_bars].eq(0.625).all()
    assert result["size"].gt(0.0).all()
    flat = strategy.indicators(data, params)
    assert np.isfinite(flat.to_numpy(dtype=float)).all()
    assert not flat["entry"].any()
    t = np.arange(700)
    close = 100.0 + 0.02 * t + 8.0 * np.sin(t / 6.0)
    data = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=len(t), freq="B"),
    )
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:430], params)
    pd.testing.assert_frame_equal(full.iloc[:430], prefix)
    assert full["entry"].any()
    full_signals = strategy.generate_signals(data, full, ctx, params).data
    prefix_signals = strategy.generate_signals(data.iloc[:430], prefix, ctx, params).data
    pd.testing.assert_frame_equal(full_signals.iloc[:430], prefix_signals)


if __name__ == "__main__":
    _self_check()
