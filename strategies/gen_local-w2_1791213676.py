from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_bars: int = 2
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791213676"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(15, int(params.streak_bars) + 6)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.streak_bars < 1 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("streak_bars must be positive and trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["valid_bar"] = (
            np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
            & data[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & data["high"].ge(data["low"])
        )
        valid_pair = out["valid_bar"] & out["valid_bar"].shift(1, fill_value=False)
        out["upper_rejection"] = (
            valid_pair & data["high"].gt(out["previous_high"])
            & data["close"].le(out["previous_high"])
        )
        out["lower_rejection"] = (
            valid_pair & data["low"].lt(out["previous_low"])
            & data["close"].ge(out["previous_low"])
        )
        for side in ("upper", "lower"):
            rejected = out[side + "_rejection"]
            out[side + "_streak"] = rejected.astype(int).groupby((~rejected).cumsum()).cumsum()
        out["channel_high"] = data["high"].rolling(5, min_periods=5).max().shift(1)
        out["channel_low"] = data["low"].rolling(5, min_periods=5).min().shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(valid_pair)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["ready"] = (
            out["valid_bar"] & np.isfinite(out["atr"]) & out["atr"].gt(0)
            & out[["channel_high", "channel_low"]].notna().all(axis=1)
        )
        out["long_entry"] = (
            out["ready"] & out["lower_streak"].shift(1).ge(params.streak_bars)
            & data["close"].gt(out["channel_high"])
        )
        out["short_entry"] = (
            out["ready"] & out["upper_streak"].shift(1).ge(params.streak_bars)
            & data["close"].lt(out["channel_low"])
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        fill_bar = -1
        for i in range(len(data)):
            if not ready[i]:
                direction = 0
                entry = np.nan
                continue
            if direction:
                if np.isnan(entry):
                    # The preceding close requested entry; this bar supplies its open.
                    entry = opens[i]
                    fill_bar = i
                    stop = -params.trail_atr * atrs[i - 1]
                    armed = False
                profit = direction * (closes[i] - entry)
                # Close-observed exits execute at the next open after the final shift.
                if profit <= stop or i - fill_bar + 1 >= 4:
                    direction = 0
                    entry = np.nan
                    continue
                armed = armed or profit >= 0.0075 * entry
                if armed:
                    # Signed profit-space stop only rises, for both longs and shorts.
                    stop = max(stop, 0.0, profit - params.trail_atr * atrs[i])
            elif longs[i] or shorts[i]:
                direction = 1 if longs[i] else -1
                entry = np.nan
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=24, freq="D")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    for direction in (1, -1):
        close = np.full(24, 100.0)
        close[16:19] = 100.0 + direction * np.array([1.0, 2.0, 0.4])
        data = pd.DataFrame({
            "open": 100.0,
            "high": np.maximum(close, 100.0) + 0.2,
            "low": np.minimum(close, 100.0) - 0.2,
            "close": close,
            "volume": 1000000.0,
        }, index=index)
        indicators = strategy.indicators(data, params)
        indicators["atr"] = 1.0
        indicators["ready"] = True
        indicators["long_entry"] = False
        indicators["short_entry"] = False
        column = "long_entry" if direction == 1 else "short_entry"
        indicators.loc[index[15], column] = True
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[15:20].tolist() == [0, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
        prefix = strategy.generate_signals(data.iloc[:18], indicators.iloc[:18], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:18])
        data["close"] = 100.0
        expiry = strategy.generate_signals(data, indicators, ctx, params).data
        assert expiry["signal"].iloc[16:21].tolist() == [direction, direction, direction, direction, 0]


if __name__ == "__main__":
    _self_check()
