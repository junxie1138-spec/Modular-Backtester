from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    range_bars: int = 5
    rank_bars: int = 63
    atr_bars: int = 14
    tail: float = 0.12
    stop_atr: float = 1.6
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791324799"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.range_bars + 1, params.atr_bars + params.rank_bars + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (
            params.range_bars >= 1
            and params.rank_bars >= 2
            and params.atr_bars >= 1
            and 0.0 < params.tail < 0.5
            and np.isfinite(params.stop_atr)
            and params.stop_atr > 0.0
            and 3 <= params.hold_bars <= 5
            and 0.0 < params.min_size <= params.max_size <= 1.0
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["range_high"] = data["high"].shift(1).rolling(params.range_bars).max()
        out["range_low"] = data["low"].shift(1).rolling(params.range_bars).min()
        out["upper_extension"] = (data["high"] - out["range_high"]).clip(lower=0.0)
        out["lower_extension"] = (out["range_low"] - data["low"]).clip(lower=0.0)
        out["extension_score"] = (
            (out["upper_extension"] - out["lower_extension"])
            / out["prior_atr"].where(out["prior_atr"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        out["percentile"] = out["extension_score"].rolling(params.rank_bars).rank(pct=True)
        out["rejected"] = (
            (data["close"] > out["range_low"])
            & (data["close"] < out["range_high"])
        )
        valid = (
            np.isfinite(out["percentile"])
            & np.isfinite(out["atr"])
            & (out["atr"] > 0.0)
            & np.isfinite(data["close"])
            & out["rejected"]
        )
        long_entry = (
            valid
            & (out["lower_extension"] > 0.0)
            & (out["upper_extension"] == 0.0)
            & (out["percentile"] <= params.tail)
        )
        short_entry = (
            valid
            & (out["upper_extension"] > 0.0)
            & (out["lower_extension"] == 0.0)
            & (out["percentile"] >= 1.0 - params.tail)
        )
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0)
        out["strength"] = (
            ((out["percentile"] - 0.5).abs() - (0.5 - params.tail)) / params.tail
        ).clip(0.0, 1.0).fillna(0.0)
        out["entry_size"] = params.min_size + (
            params.max_size - params.min_size
        ) * out["strength"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, params.min_size, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        stop_distance = 0.0
        fixed_stop = np.nan
        position_size = params.min_size

        for i in range(n):
            if side != 0:
                if i == entry_bar:
                    # Anchor once to the execution bar's open, using decision-bar ATR.
                    fixed_stop = opens[i] - side * stop_distance
                stopped = (
                    not np.isfinite(closes[i])
                    or not np.isfinite(fixed_stop)
                    or side * (closes[i] - fixed_stop) < 0.0
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                    continue
            elif (
                candidates[i] != 0
                and np.isfinite(atr[i])
                and atr[i] > 0.0
                and np.isfinite(entry_sizes[i])
            ):
                side = int(candidates[i])
                entry_bar = i + 1
                stop_distance = params.stop_atr * atr[i]
                position_size = float(entry_sizes[i])
            raw_signal[i] = side
            if side != 0:
                raw_size[i] = position_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(raw_size, index=data.index).shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams(stop_atr=2.0)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
        indicators = pd.DataFrame({"candidate": 0, "atr": 1.0, "entry_size": 0.7}, index=index)
        indicators.iloc[0, 0] = direction
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0]
        assert np.allclose(result.loc[result["signal"] != 0, "size"], 0.7)
        # Favorable movement must not ratchet the fixed stop.
        data["close"] = 100.0 + direction * np.array([0.0, 3.0, -1.0, -3.0, 0.0, 0.0, 0.0])
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0, 0]
        assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
