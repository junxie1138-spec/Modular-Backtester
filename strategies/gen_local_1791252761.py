from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    visibility_span: int = 8
    topology_window: int = 20
    alignment_min: float = 0.35
    positive_share_min: float = 0.60
    atr_period: int = 14
    atr_multiple: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791252761"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(200, params.visibility_span + params.topology_window + 1, params.atr_period + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.visibility_span < 2 or params.topology_window < 2 or params.atr_period < 1:
            raise ValueError("Indicator windows must be positive; graph windows must be at least two.")
        close = data["close"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        count = pd.Series(0.0, index=data.index)
        products = count.copy()
        energy = count.copy()
        positive = count.copy()
        # Each edge ends on the current bar; no future endpoint is inspected.
        for lag in range(2, params.visibility_span + 1):
            earlier = returns.shift(lag)
            interior_high = returns.shift(1).rolling(lag - 1, min_periods=lag - 1).max()
            visible = interior_high.lt(pd.concat([returns, earlier], axis=1).min(axis=1))
            visible &= returns.notna() & earlier.notna()
            product = (returns * earlier).where(visible, 0.0).fillna(0.0)
            count += visible.astype(float)
            products += product
            energy += product.abs()
            positive += (visible & returns.gt(0) & earlier.gt(0)).astype(float)
        window = params.topology_window
        edge_count = count.rolling(window, min_periods=window).sum()
        edge_product = products.rolling(window, min_periods=window).sum()
        edge_energy = energy.rolling(window, min_periods=window).sum()
        positive_edges = positive.rolling(window, min_periods=window).sum()
        return pd.DataFrame({
            "return": returns,
            "ma200": close.rolling(200, min_periods=200).mean(),
            "atr": true_range.rolling(params.atr_period, min_periods=params.atr_period).mean(),
            "edge_count": edge_count,
            "edge_product": edge_product,
            "edge_energy": edge_energy,
            "positive_edges": positive_edges,
            "alignment": edge_product.div(edge_energy.where(edge_energy.gt(0))),
            "positive_share": positive_edges.div(edge_count.where(edge_count.gt(0))),
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        if params.atr_multiple <= 0 or params.max_hold < 1:
            raise ValueError("ATR multiple and holding period must be positive.")
        ready = indicators[["return", "ma200", "atr", "alignment", "positive_share", "edge_count"]].notna().all(axis=1)
        setup = (
            ready
            & data["close"].gt(indicators["ma200"])
            & indicators["return"].gt(0)
            & indicators["atr"].gt(0)
            & indicators["edge_count"].ge(params.topology_window)
            & indicators["alignment"].ge(params.alignment_min)
            & indicators["positive_share"].ge(params.positive_share_min)
        ).fillna(False)
        entry = setup & ~setup.shift(1, fill_value=False)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        regimes = data["close"].gt(indicators["ma200"]).to_numpy(dtype=bool)
        entries = entry.to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if active:
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    active = False
                else:
                    # The first in-trade close is the bar after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                    if close <= stop or not regimes[i] or i - entry_bar >= params.max_hold:
                        active = False
            elif entries[i]:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100.0, 102.0, 104.0, 103.0, 101.0, 100.0, 100.0]}, index=index)
    indicators = pd.DataFrame({
        "return": [0.01, 0.02, 0.02, -0.01, -0.02, -0.01, 0.0],
        "ma200": 90.0,
        "atr": [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
        "edge_count": 20.0,
        "alignment": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "positive_share": 1.0,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, VisibilityParams()).data
    # Expanding ATR cannot loosen the stop; the exit fills one bar later.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(VisibilityParams()) >= 200


if __name__ == "__main__":
    _self_check()
