from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DiffusionParams:
    fractional_order: float = 0.4
    memory_bars: int = 40
    diffusion_bars: int = 12
    transmission_ratio: float = 1.6
    profit_target: float = 0.025
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[DiffusionParams]):
    strategy_id = "gen_local-w2_1791258972"

    @classmethod
    def params_type(cls):
        return DiffusionParams

    def warmup_bars(self, params: DiffusionParams) -> int:
        return params.memory_bars + params.diffusion_bars + 1

    def indicators(self, data: pd.DataFrame, params: DiffusionParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if params.memory_bars < 2 or params.diffusion_bars < 2:
            raise ValueError("lookbacks must be at least two")
        if params.transmission_ratio <= 1.0 or params.profit_target <= 0.0 or params.holding_bars < 1:
            raise ValueError("invalid transmission ratio, target, or holding period")
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0.0))
        weights = np.ones(params.memory_bars, dtype=float)
        for k in range(1, params.memory_bars):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1.0) / k
        # ponytail: fixed finite memory; increase memory_bars if truncation matters.
        fractional_close = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fractional_close = fractional_close + weight * close.shift(k)
        step = fractional_close.diff()
        window = params.diffusion_bars
        positive = step.clip(lower=0.0).rolling(window, min_periods=window).sum()
        negative = (-step.clip(upper=0.0)).rolling(window, min_periods=window).sum()
        valid = step.notna()
        positive_share = (step > 0.0).astype(float).where(valid).rolling(window, min_periods=window).mean()
        negative_share = (step < 0.0).astype(float).where(valid).rolling(window, min_periods=window).mean()
        total = (positive + negative).replace(0.0, np.nan)
        pressure = (positive - negative) / total
        threshold = (params.transmission_ratio - 1.0) / (params.transmission_ratio + 1.0)
        long_entry = (step > 0.0) & (pressure > threshold) & positive_share.between(0.5, 0.8)
        short_entry = (step < 0.0) & (pressure < -threshold) & negative_share.between(0.5, 0.8)
        candidate = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index, dtype=int)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_step": step,
            "positive_mass": positive,
            "negative_mass": negative,
            "positive_share": positive_share,
            "negative_share": negative_share,
            "pressure": pressure,
            "candidate": candidate,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DiffusionParams) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        first_decision = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if position:
                # Yesterday's decision enters at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and np.isfinite(closes[i]) else np.nan
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= params.holding_bars:
                    position = 0
                    # A flat decision prevents same-close exit and re-entry.
            elif i >= first_decision and i + 1 < len(data) and np.isfinite(closes[i]) and closes[i] > 0.0:
                if confirmed[i]:
                    position = int(confirmed[i])
                    entry_bar = i + 1
                    entry_price = np.nan
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DiffusionParams(memory_bars=2, diffusion_bars=2, holding_bars=2, profit_target=0.02)
    index = pd.date_range("2020-01-01", periods=9)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for direction, entry_close, exit_bar in [(1, 103.0, 6), (-1, 97.0, 6), (1, 100.0, 7)]:
        data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
        data.iloc[5, data.columns.get_loc("close")] = entry_close
        indicators = pd.DataFrame({"candidate": [0, 0, 0, direction, direction, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[:5].eq(0).all()
        assert result["signal"].iloc[5:exit_bar].eq(direction).all()
        assert result["signal"].iloc[exit_bar:].eq(0).all()
        assert result["size"].gt(0.0).all()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:6], params)
    pd.testing.assert_frame_equal(full.iloc[:6], prefix)
