from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    pattern_window: int = 24
    complexity_min: float = 0.04
    drift_window: int = 8
    efficiency_min: float = 0.35
    hold_bars: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791280988"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.pattern_window + 2, params.drift_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.pattern_window < 2 or params.drift_window < 1:
            raise ValueError("Lookbacks must be positive; pattern_window must be at least two.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        if not 0 <= params.complexity_min <= 1 or not 0 <= params.efficiency_min <= 1:
            raise ValueError("Thresholds must be between zero and one.")
        if not np.isfinite(params.size) or params.size <= 0:
            raise ValueError("size must be finite and positive.")

        out = pd.DataFrame(index=data.index)
        volume = data["volume"].astype(float).where(lambda s: np.isfinite(s) & (s > 0))
        out["volume_lag1"] = volume.shift(1)
        out["volume_lag2"] = volume.shift(2)
        a, b, c = out["volume_lag2"], out["volume_lag1"], volume
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        patterns = [
            (a < b) & (b < c), (a < c) & (c < b),
            (b < a) & (a < c), (b < c) & (c < a),
            (c < a) & (a < b), (c < b) & (b < a),
        ]
        out["volume_pattern"] = pd.Series(
            np.select(patterns, np.arange(6), default=np.nan), index=data.index
        ).where(valid)
        probability_columns = []
        for code in range(6):
            name = f"pattern_probability_{code}"
            out[name] = out["volume_pattern"].eq(code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            probability_columns.append(name)

        p = out[probability_columns]
        entropy = -(p * np.log(p.where(p > 0, 1.0))).sum(axis=1, min_count=6)
        mixture = (p + 1.0 / 6.0) / 2.0
        mixture_entropy = -(mixture * np.log(mixture)).sum(axis=1, min_count=6)
        # Entropy times disequilibrium: ordered and random participation both score low.
        out["ordinal_entropy"] = entropy / np.log(6.0)
        out["ordinal_disequilibrium"] = (
            mixture_entropy - 0.5 * entropy - 0.5 * np.log(6.0)
        ).clip(lower=0) / np.log(2.0)
        out["ordinal_complexity"] = out["ordinal_entropy"] * out["ordinal_disequilibrium"]
        close = data["close"].astype(float).where(lambda s: np.isfinite(s) & (s > 0))
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["price_change"] = close.diff()
        out["drift"] = close - close.shift(params.drift_window)
        out["path_length"] = out["price_change"].abs().rolling(
            params.drift_window, min_periods=params.drift_window
        ).sum()
        out["efficiency"] = out["drift"].abs() / out["path_length"].replace(0.0, np.nan)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2) or not np.isfinite(params.size) or params.size <= 0:
            raise ValueError("Require hold_bars in (1, 2) and finite positive size.")
        ready = np.isfinite(indicators[[
            "ordinal_complexity", "efficiency", "drift", "ma200"
        ]]).all(axis=1)
        active = ready & indicators["ordinal_complexity"].ge(params.complexity_min)
        active &= indicators["efficiency"].ge(params.efficiency_min)
        long_entry = active & indicators["drift"].gt(0) & data["close"].gt(indicators["ma200"])
        short_entry = active & indicators["drift"].lt(0) & data["close"].lt(indicators["ma200"])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        position, exit_bar = 0, -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    # Keep this decision flat; re-entry can occur on the next bar.
                else:
                    raw[i] = position
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
                raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [110, 90, 90, 90, 110, 110, 110]}, index=index)
    indicators = pd.DataFrame({
        "ordinal_complexity": 0.25, "efficiency": 1.0, "ma200": 100.0,
        "drift": [1, -1, -1, -1, 1, 1, 1],
    }, index=index)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert signals["size"].gt(0).all()
    flat = pd.DataFrame(100.0, index=pd.date_range("2020-01-01", periods=230),
                        columns=["open", "high", "low", "close", "volume"])
    flat_indicators = strategy.indicators(flat, params)
    assert strategy.warmup_bars(params) >= 200
    assert strategy.generate_signals(flat, flat_indicators, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
