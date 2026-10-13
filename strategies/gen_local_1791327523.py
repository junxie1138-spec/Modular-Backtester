from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    graph_window: int = 12
    evidence_days: int = 4
    minimum_rebound: float = 0.006
    minimum_drawdown: float = 0.003
    maximum_drawdown: float = 0.08


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791327523"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(200, max(4, int(params.graph_window)) + max(1, int(params.evidence_days)) - 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        window = max(4, int(params.graph_window))
        memory = max(1, int(params.evidence_days))
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        prices = np.column_stack([
            log_close.shift(window - 1 - j).to_numpy()
            for j in range(window)
        ])
        valid = np.isfinite(prices).all(axis=1)
        safe_prices = np.where(np.isfinite(prices), prices, 0.0)
        rows = np.arange(len(data))
        # Choose the most recent maximum as the drawdown's anchor.
        peak = window - 1 - np.argmax(safe_prices[:, ::-1], axis=1)
        distance = np.full((len(data), window), np.inf)
        distance[rows, peak] = 0.0

        # Edges require every intervening close to lie below the sight line.
        # Only chronological paths are allowed: each hop advances toward today.
        for right in range(1, window):
            for left in range(right):
                visible = valid & (peak <= left)
                for middle in range(left + 1, right):
                    sight_line = safe_prices[:, left] + (
                        safe_prices[:, right] - safe_prices[:, left]
                    ) * ((middle - left) / (right - left))
                    visible &= safe_prices[:, middle] < sight_line
                candidate = np.where(visible, distance[:, left] + 1.0, np.inf)
                distance[:, right] = np.minimum(distance[:, right], candidate)

        hops = np.where(valid, distance[:, -1], np.nan)
        previous_hops = np.where(valid, distance[:, -2], np.nan)
        contraction = np.zeros(len(data), dtype=bool)
        eligible = valid & (peak <= window - 4)
        contraction[eligible] = (
            previous_hops[eligible] - hops[eligible]
        ) >= 1.0

        out = pd.DataFrame(index=data.index)
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["drawdown"] = close / close.rolling(window, min_periods=window).max() - 1.0
        out["rebound"] = close / close.rolling(window, min_periods=window).min().replace(0, np.nan) - 1.0
        out["graph_hops"] = hops
        out["previous_graph_hops"] = previous_hops
        out["path_contraction"] = pd.Series(contraction, index=data.index).astype(float)
        out["recent_contraction"] = out["path_contraction"].rolling(memory, min_periods=memory).max()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        condition = (
            data["close"].ge(indicators["regime_ma"])
            & indicators["drawdown"].le(-params.minimum_drawdown)
            & indicators["drawdown"].ge(-params.maximum_drawdown)
            & indicators["rebound"].ge(params.minimum_rebound)
            & indicators["recent_contraction"].ge(1.0)
        ).fillna(False)
        # The same condition controls entry and signal-reversal exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = condition.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    close = 100.0 + np.arange(240) * 0.1
    data = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not np.isinf(derived.to_numpy()).any()
    assert strategy.warmup_bars(params) >= 200
    sample = data.iloc[:3]
    supplied = pd.DataFrame({
        "regime_ma": [90.0, 90.0, 90.0],
        "drawdown": [-0.02, -0.02, -0.02],
        "rebound": [0.01, 0.01, 0.01],
        "recent_contraction": [1.0, 0.0, 0.0],
    }, index=sample.index)
    result = strategy.generate_signals(sample, supplied, None, params).data
    assert result["signal"].tolist() == [0, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
