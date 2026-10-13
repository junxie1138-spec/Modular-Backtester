from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalOrdinalParams:
    entropy_window: int = 21
    entropy_split: float = 0.90
    edge_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[SeasonalOrdinalParams]):
    strategy_id = "gen_local_1790852245"

    @classmethod
    def params_type(cls) -> type[SeasonalOrdinalParams]:
        return SeasonalOrdinalParams

    def warmup_bars(self, params: SeasonalOrdinalParams) -> int:
        return 756 + 18 + params.entropy_window + 3 + 4

    def indicators(self, data: pd.DataFrame, params: SeasonalOrdinalParams) -> pd.DataFrame:
        if params.entropy_window < 3:
            raise ValueError("entropy_window must be at least 3")
        if not 0.0 < params.entropy_split < 1.0:
            raise ValueError("entropy_split must be between 0 and 1")
        if not 0.0 < params.edge_threshold < 1.0:
            raise ValueError("edge_threshold must be between 0 and 1")

        features = {}
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns use chronological order as a deterministic tie break.
        code = ((a > b).astype(int) * 4 + (a > c).astype(int) * 2
                + (b > c).astype(int)).astype(float).where(valid)
        features["return"] = returns
        features["return_lag2"] = a
        features["return_lag1"] = b
        features["ordinal_code"] = code
        entropy = pd.Series(0.0, index=data.index)
        for pattern in (0, 1, 3, 4, 6, 7):
            frequency = code.eq(pattern).astype(float).where(valid).rolling(
                params.entropy_window, min_periods=params.entropy_window
            ).mean()
            features[f"pattern_probability_{pattern}"] = frequency
            entropy = entropy - frequency * np.log(frequency.where(frequency > 0, 1.0))
        entropy = entropy / np.log(6.0)
        features["entropy"] = entropy
        regime = entropy.ge(params.entropy_split).astype(float).where(entropy.notna())
        month = pd.Series(data.index.month, index=data.index, dtype=float)
        features["regime"] = regime
        features["month"] = month

        # Each observation joins its origin month/regime only after 18 bars mature.
        outcome = data["close"].pct_change(18, fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        origin_month = month.shift(18)
        origin_regime = regime.shift(18)
        features["matured_return"] = outcome
        features["origin_month"] = origin_month
        features["origin_regime"] = origin_regime
        raw_edge = pd.Series(np.nan, index=data.index, dtype=float)
        for calendar_month in range(1, 13):
            for complexity_regime in (0, 1):
                matched = (origin_month.eq(calendar_month)
                           & origin_regime.eq(complexity_regime) & outcome.notna())
                count = matched.astype(float).rolling(756, min_periods=756).sum()
                total = outcome.where(matched, 0.0).rolling(756, min_periods=756).sum()
                absolute = outcome.abs().where(matched, 0.0).rolling(
                    756, min_periods=756
                ).sum()
                edge = (total / absolute.where(absolute > 0)).where(count >= 12)
                key = f"month_{calendar_month}_regime_{complexity_regime}"
                features[key + "_count"] = count
                features[key + "_return_sum"] = total
                features[key + "_absolute_sum"] = absolute
                features[key + "_edge"] = edge
                current = month.eq(calendar_month) & regime.eq(complexity_regime)
                raw_edge = raw_edge.where(~current, edge)
        features["raw_seasonal_edge"] = raw_edge
        features["seasonal_edge"] = raw_edge.rolling(5, min_periods=5).mean()
        return pd.DataFrame(features, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalOrdinalParams,
    ) -> SignalFrame:
        state = 0
        positions = np.zeros(len(data), dtype=int)
        for i, edge in enumerate(indicators["seasonal_edge"].to_numpy()):
            # Mirrored thresholds: positive evidence enters, negative evidence exits.
            # Missing estimates and the neutral band preserve the existing position.
            if np.isfinite(edge):
                if edge >= params.edge_threshold:
                    state = 1
                elif edge <= -params.edge_threshold:
                    state = 0
            positions[i] = state
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SeasonalOrdinalParams()
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2015-01-02", periods=1100)
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index))))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:950], params)
    pd.testing.assert_frame_equal(full.iloc[:950], prefix)
    edges = pd.DataFrame({
        "seasonal_edge": [0.2, 0.0, np.nan, -0.2, 0.0, 0.2]
    }, index=index[:6])
    signals = strategy.generate_signals(data.iloc[:6], edges, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert signals["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 756 + 18 + params.entropy_window + 3 + 4


if __name__ == "__main__":
    _self_check()
