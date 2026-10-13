from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticRegimeParams:
    regime_window: int = 63
    block_bars: int = 5
    stretch_window: int = 20
    regime_band: float = 0.20
    stretch_threshold: float = 1.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ElasticRegimeParams]):
    strategy_id = "gen_local_1790963890"

    @classmethod
    def params_type(cls):
        return ElasticRegimeParams

    def warmup_bars(self, params: ElasticRegimeParams) -> int:
        return int(max(params.regime_window + params.block_bars,
                       params.stretch_window + 1) + 1)

    def indicators(self, data: pd.DataFrame,
                   params: ElasticRegimeParams) -> pd.DataFrame:
        if (params.regime_window < 2 or params.block_bars < 2
                or params.stretch_window < 2 or params.holding_bars < 1
                or not 0.0 < params.regime_band < 1.0
                or not np.isfinite(params.stretch_threshold)
                or params.stretch_threshold <= 0.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["block_return"] = out["return"].rolling(params.block_bars).sum()
        out["daily_variance"] = out["return"].rolling(
            params.regime_window).var(ddof=0)
        out["block_variance"] = out["block_return"].rolling(
            params.regime_window).var(ddof=0)
        variance = out["daily_variance"].where(out["daily_variance"] > 1e-16)
        out["variance_ratio"] = out["block_variance"] / (
            params.block_bars * variance)
        out["stretch_return"] = out["return"].rolling(
            params.stretch_window).sum()
        out["stretch"] = out["stretch_return"] / np.sqrt(
            params.stretch_window * variance)
        out["regime"] = np.select(
            [out["variance_ratio"] > 1.0 + params.regime_band,
             out["variance_ratio"] < 1.0 - params.regime_band],
            [1, -1], default=0).astype(int)
        direction = np.sign(out["stretch"].fillna(0.0)).astype(int)
        eligible = out["stretch"].abs().ge(params.stretch_threshold)
        out["candidate"] = (direction * out["regime"]).where(
            eligible, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: ElasticRegimeParams) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        regime = indicators["regime"].fillna(0).astype(int)
        confirmed = candidate.where(
            candidate.ne(0) & candidate.eq(candidate.shift(1))
            & regime.eq(regime.shift(1)), 0).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(confirmed):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ElasticRegimeParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    synthetic = pd.DataFrame({"candidate": -1, "regime": -1}, index=index)
    synthetic.iloc[:3] = 1
    result = strategy.generate_signals(data, synthetic, ctx, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20] == 0
    assert result["signal"].iloc[21:39].eq(-1).all()
    assert result["signal"].iloc[39] == 0
    assert result["size"].gt(0).all()
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, ctx, params).data[
        "signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
