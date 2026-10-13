from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathAreaParams:
    area_threshold: float = 0.06
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[PathAreaParams]):
    strategy_id = "gen_local_1791314216"

    @classmethod
    def params_type(cls):
        return PathAreaParams

    def warmup_bars(self, params: PathAreaParams) -> int:
        return 49

    def indicators(self, data: pd.DataFrame, params: PathAreaParams) -> pd.DataFrame:
        if not (np.isfinite(params.area_threshold) and 0 < params.area_threshold < 1):
            raise ValueError("area_threshold must be finite and between zero and one")
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["channel_low"] = data["low"].rolling(40, min_periods=40).min()
        out["channel_high"] = data["high"].rolling(40, min_periods=40).max()
        out["channel_width"] = (out["channel_high"] - out["channel_low"]).replace(0, np.nan)
        out["position"] = ((data["close"] - out["channel_low"]) / out["channel_width"]).clip(0, 1)
        out["position_change"] = out["position"].diff()
        out["path_start"] = out["position"].shift(9)
        out["path_sum"] = out["position"].rolling(10, min_periods=10).sum()
        # Half of the antisymmetric level-two signature for (normalized time, position).
        # Translating the path to its starting point gives this exact polygon area.
        out["signature_area"] = (5 * (out["path_start"] + out["position"]) - out["path_sum"]) / 9
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: PathAreaParams,
    ) -> SignalFrame:
        position = indicators["position"]
        area = indicators["signature_area"]
        change = indicators["position_change"]
        ready = np.isfinite(indicators[["position", "signature_area", "position_change", "atr"]]).all(axis=1)
        ready &= indicators["atr"] > 0
        longs = ready & (position < 0.35) & (area > params.area_threshold) & (change > 0)
        shorts = ready & (position > 0.65) & (area < -params.area_threshold) & (change < 0)
        candidates = np.select([longs, shorts], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side, entry_bar = 0, -1
        stop = np.nan
        for i in range(len(data)):
            if side:
                if entry_bar == -1:
                    # Yesterday's decision fills at today's open; freeze yesterday's ATR.
                    entry_bar = i
                    stop = opens[i] - side * params.stop_atr * atr[i - 1]
                invalid = not (np.isfinite(stop) and np.isfinite(closes[i]) and opens[entry_bar] > 0)
                stopped = (side == 1 and closes[i] < stop) or (side == -1 and closes[i] > stop)
                if invalid or stopped or i - entry_bar + 1 >= 18:
                    side, entry_bar = 0, -1
                    stop = np.nan
            elif candidates[i]:
                side = int(candidates[i])
                entry_bar = -1
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = PathAreaParams()
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=49)
    for side, hit in [(1, 94.0), (-1, 106.0)]:
        ind = pd.DataFrame(
            {"position": np.nan, "signature_area": np.nan, "position_change": np.nan, "atr": 2.0},
            index=index,
        )
        ind.iloc[0] = [0.2 if side == 1 else 0.8, side * 0.1, side * 0.01, 2.0]
        baseline = strategy.generate_signals(data, ind, ctx, params).data
        assert baseline["signal"].iloc[0] == 0
        assert (baseline["signal"].iloc[1:19] == side).all()
        assert (baseline["signal"].iloc[19:] == 0).all()
        moved = data.copy()
        moved.loc[index[2], "close"] = 110.0 if side == 1 else 90.0
        moved.loc[index[3], "close"] = 99.0 if side == 1 else 101.0
        moved.loc[index[4], "close"] = hit
        result = strategy.generate_signals(moved, ind, ctx, params).data
        assert result["signal"].iloc[4] == side
        assert (result["signal"].iloc[5:] == 0).all()
        assert (result["size"] > 0).all()
        prefix = strategy.generate_signals(moved.iloc[:5], ind.iloc[:5], ctx, params).data
        assert prefix.equals(result.iloc[:5])
    long_data = pd.concat([data] * 3, ignore_index=True)
    long_data.index = pd.date_range("2020-01-01", periods=len(long_data))
    derived = strategy.indicators(long_data, params)
    assert derived["signature_area"].first_valid_index() == long_data.index[48]
    assert strategy.warmup_bars(params) >= 49
    assert strategy.generate_signals(long_data, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
