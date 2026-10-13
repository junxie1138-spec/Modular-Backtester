from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    window: int = 18
    threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1791244089"

    @classmethod
    def params_type(cls) -> type[TopologyParams]:
        return TopologyParams

    @staticmethod
    def warmup_bars(params: TopologyParams) -> int:
        return 3 * params.window

    @staticmethod
    def indicators(data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if params.window < 3 or not 0.0 < params.threshold < 1.0:
            raise ValueError("window must be >= 3; threshold must be between 0 and 1")
        w = params.window
        close = data["close"].astype(float)
        log_close = np.log(close.where(np.isfinite(close) & (close > 0.0)))
        x = log_close.to_numpy()
        upper = log_close.shift(1).to_numpy() - x
        lower = upper.copy()
        valley_mass = np.zeros(len(data), dtype=float)
        crest_mass = np.zeros(len(data), dtype=float)
        # Natural visibility: a lag-d edge clears every intervening backward slope.
        # Reflected prices reverse the inequality; long edges measure bowl versus dome geometry.
        for lag in range(2, w + 1):
            slope = (log_close.shift(lag).to_numpy() - x) / lag
            if lag >= max(2, w // 3):
                valley_mass += lag * (slope > upper)
                crest_mass += lag * (slope < lower)
            upper = np.maximum(upper, slope)
            lower = np.minimum(lower, slope)
        valid = log_close.rolling(w + 1, min_periods=w + 1).count().eq(w + 1)
        valley = pd.Series(valley_mass, index=data.index).where(valid)
        crest = pd.Series(crest_mass, index=data.index).where(valid)
        valley_total = valley.rolling(w, min_periods=w).sum()
        crest_total = crest.rolling(w, min_periods=w).sum()
        total = valley_total + crest_total
        topology = ((valley_total - crest_total) / total.replace(0.0, np.nan)).fillna(0.0)
        anchor = log_close.rolling(3 * w, min_periods=3 * w).mean()
        return pd.DataFrame({
            "log_close": log_close,
            "valley_mass": valley,
            "crest_mass": crest,
            "valley_total": valley_total,
            "crest_total": crest_total,
            "topology": topology,
            "anchor": anchor,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        # The same condition maintains the position and reverses it to flat.
        eligible = (
            (indicators["topology"] > params.threshold)
            & (indicators["log_close"] < indicators["anchor"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = eligible.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = TopologyParams(window=6)
    index = pd.date_range("2020-01-01", periods=90, freq="B")
    close = np.exp(0.0001 * (np.arange(90) - 45.0) ** 2)
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=index)
    ind = GeneratedStrategy.indicators(data, params)
    assert ind["topology"].iloc[GeneratedStrategy.warmup_bars(params):].eq(1.0).all()
    reflected = data.copy()
    reflected["close"] = 1.0 / close
    assert GeneratedStrategy.indicators(reflected, params)["topology"].iloc[18:].eq(-1.0).all()
    prefix = GeneratedStrategy.indicators(data.iloc[:60], params)
    pd.testing.assert_frame_equal(prefix, ind.iloc[:60])
    signals = GeneratedStrategy.generate_signals(data, ind, None, params).data
    expected = ((ind["topology"] > params.threshold) & (ind["log_close"] < ind["anchor"])).astype(int)
    pd.testing.assert_series_equal(signals["signal"], expected.shift(1).fillna(0).astype(int), check_names=False)
    assert signals["signal"].eq(1).any() and signals["signal"].iloc[-1] == 0
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
