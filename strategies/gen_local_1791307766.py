from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalBreakoutParams:
    pattern_window: int = 18
    channel_window: int = 12
    complexity_floor: float = 0.025
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalBreakoutParams]):
    strategy_id = "gen_local_1791307766"

    @classmethod
    def params_type(cls):
        return OrdinalBreakoutParams

    @staticmethod
    def warmup_bars(params: OrdinalBreakoutParams) -> int:
        return int(max(params.pattern_window + 4, params.channel_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalBreakoutParams) -> pd.DataFrame:
        if params.pattern_window < 3 or params.channel_window < 2:
            raise ValueError("Lookback windows are too small.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        if not 0.0 <= params.complexity_floor <= 1.0:
            raise ValueError("complexity_floor must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        triples = np.column_stack([
            out["log_return"].shift(2),
            out["log_return"].shift(1),
            out["log_return"],
        ])
        valid = np.isfinite(triples).all(axis=1)
        # Stable ordering breaks equal-return ties chronologically.
        orders = np.argsort(triples, axis=1, kind="stable")
        codes = orders @ np.array([1, 3, 9])
        out["ordinal_code"] = np.where(valid, codes, np.nan)
        pattern_codes = (21, 15, 19, 7, 11, 5)
        names = []
        for code in pattern_codes:
            name = f"frequency_{code}"
            names.append(name)
            observations = (out["ordinal_code"] == code).astype(float).where(valid)
            out[name] = observations.rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
        probabilities = out[names]
        out["entropy"] = -(
            probabilities * np.log(probabilities.clip(lower=1e-12))
        ).sum(axis=1, min_count=6) / np.log(6.0)
        out["disequilibrium"] = (
            (probabilities - 1.0 / 6.0) ** 2
        ).sum(axis=1, min_count=6) / (5.0 / 6.0)
        out["complexity"] = out["entropy"] * out["disequilibrium"]
        out["ordinal_vote"] = out["frequency_21"] - out["frequency_5"]
        out["setup_complexity"] = out["complexity"].shift(1)
        out["setup_vote"] = out["ordinal_vote"].shift(1)
        out["upper_channel"] = data["high"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).max().shift(1)
        out["lower_channel"] = data["low"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).min().shift(1)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalBreakoutParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        ready = np.isfinite(indicators[[
            "setup_complexity", "setup_vote", "upper_channel", "lower_channel"
        ]]).all(axis=1) & np.isfinite(data["close"])
        complex_setup = ready & (
            indicators["setup_complexity"] >= params.complexity_floor
        )
        long_entry = complex_setup & (indicators["setup_vote"] > 0) & (
            data["close"] > indicators["upper_channel"]
        )
        short_entry = complex_setup & (indicators["setup_vote"] < 0) & (
            data["close"] < indicators["lower_channel"]
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
                else:
                    signals[i] = position
            elif entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = OrdinalBreakoutParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"close": 2.0}, index=index)
    indicator_frame = pd.DataFrame({
        "setup_complexity": 0.1, "setup_vote": 1.0,
        "upper_channel": 1.0, "lower_channel": 3.0,
    }, index=index)
    for direction in (1, -1):
        indicator_frame["setup_vote"] = float(direction)
        result = GeneratedStrategy.generate_signals(
            data, indicator_frame, None, params
        ).data
        assert result["signal"].tolist() == [
            0, direction, direction, direction, direction, 0,
            direction, direction, direction, direction, 0, direction
        ]
        assert (result["size"] > 0).all()
    indicator_frame["setup_complexity"] = np.nan
    assert not GeneratedStrategy.generate_signals(
        data, indicator_frame, None, params
    ).data["signal"].any()
