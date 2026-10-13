from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureCompressionParams:
    compression_bars: int = 12
    holding_bars: int = 7

    def __post_init__(self):
        if self.compression_bars < 2 or self.holding_bars < 1:
            raise ValueError("compression_bars must be >= 2 and holding_bars >= 1")


class GeneratedStrategy(BaseStrategy[SignatureCompressionParams]):
    strategy_id = "gen_local-w2_1791221567"

    @classmethod
    def params_type(cls):
        return SignatureCompressionParams

    def warmup_bars(self, params):
        return int(4 * params.compression_bars + 1)

    def indicators(self, data, params):
        n = params.compression_bars
        close = data["close"].where(data["close"] > 0)
        spread = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        dx = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        dy = ((2.0 * close - data["high"] - data["low"]) / spread.replace(0.0, np.nan)).clip(-1.0, 1.0)
        dy = dy.mask(spread.eq(0.0), 0.0)
        relative_range = spread / close
        recent_range = relative_range.rolling(n, min_periods=n).mean()
        baseline_range = relative_range.rolling(3 * n, min_periods=3 * n).mean().shift(n)
        compression = recent_range / baseline_range.replace(0.0, np.nan)

        # Antisymmetric second-order signature: half the sum of ordered cross-products.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, n):
            cross = dx.shift(lag) * dy - dy.shift(lag) * dx
            area = area + 0.5 * cross.rolling(n - lag, min_periods=n - lag).sum()
        scale = dx.abs().rolling(n, min_periods=n).sum() * dy.abs().rolling(n, min_periods=n).sum()
        normalized_area = area / scale.replace(0.0, np.nan)
        pressure = dy.rolling(n, min_periods=n).mean()
        compressed = compression.lt(0.8)
        buy = compressed & normalized_area.lt(-0.06) & pressure.gt(0.15)
        sell = compressed & normalized_area.gt(0.06) & pressure.lt(-0.15)

        out = pd.DataFrame(index=data.index)
        out["price_increment"] = dx
        out["pressure_increment"] = dy
        out["relative_range"] = relative_range
        out["recent_range"] = recent_range
        out["baseline_range"] = baseline_range
        out["compression"] = compression.fillna(1.0)
        out["signature_area"] = area
        out["signature_scale"] = scale
        out["normalized_area"] = normalized_area
        out["mean_pressure"] = pressure
        out["entry"] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data, indicators, ctx, params):
        entries = indicators["entry"].reindex(data.index).fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0 and i >= exit_bar:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SignatureCompressionParams(compression_bars=8, holding_bars=5)
    index = pd.date_range("2020-01-01", periods=14, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [1] + [-1] * 13}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0, -1]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()
