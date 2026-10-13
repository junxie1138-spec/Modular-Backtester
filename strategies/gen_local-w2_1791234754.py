from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTrapParams:
    volume_multiple: float = 1.5
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[GapTrapParams]):
    strategy_id = "gen_local-w2_1791234754"

    @classmethod
    def params_type(cls):
        return GapTrapParams

    @staticmethod
    def warmup_bars(params: GapTrapParams) -> int:
        return 22

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapTrapParams) -> pd.DataFrame:
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 1:
            raise ValueError("volume_multiple must exceed one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["volume_baseline"] = data["volume"].rolling(20, min_periods=20).mean().shift(1)
        out["volume_ratio"] = data["volume"].div(out["volume_baseline"].where(out["volume_baseline"] > 0))
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["trap_upper"] = data["high"].shift(2)
        out["trap_lower"] = data["low"].shift(2)
        out["quiet_gap_up"] = (
            (data["open"] > out["previous_high"])
            & (data["close"] > out["previous_high"])
            & (out["volume_ratio"] < 1.0)
            & (data["volume"] > 0)
        )
        out["quiet_gap_down"] = (
            (data["open"] < out["previous_low"])
            & (data["close"] < out["previous_low"])
            & (out["volume_ratio"] < 1.0)
            & (data["volume"] > 0)
        )
        out["prior_gap_up"] = out["quiet_gap_up"].shift(1, fill_value=False)
        out["prior_gap_down"] = out["quiet_gap_down"].shift(1, fill_value=False)
        confirmed = out["volume_ratio"] >= params.volume_multiple
        valid = (
            np.isfinite(out["atr"]) & (out["atr"] > 0)
            & np.isfinite(out["volume_ratio"])
            & np.isfinite(out["trap_upper"])
            & np.isfinite(out["trap_lower"])
        )
        out["long_entry"] = (
            valid & confirmed & out["prior_gap_down"]
            & (data["close"] > out["trap_lower"])
            & (data["close"] > data["open"])
        ).fillna(False)
        out["short_entry"] = (
            valid & confirmed & out["prior_gap_up"]
            & (data["close"] < out["trap_upper"])
            & (data["close"] < data["open"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTrapParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry = stop = peak = np.nan
        armed = False
        entry_bar = -1
        for i, (opening, high, low, close) in enumerate(prices):
            if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0:
                side, entry, armed = 0, np.nan, False
                continue
            if side:
                if not np.isfinite(entry):
                    # The preceding close's decision fills at today's open.
                    entry = side * opening
                    stop = entry - params.trail_atr * atrs[i - 1]
                    peak = entry
                    entry_bar = i
                # Signed prices make the protective stop increase for both sides.
                favorable = side * (high if side == 1 else low)
                adverse = side * (low if side == 1 else high)
                # Test the previously established stop before updating it.
                exited = adverse <= stop or i - entry_bar + 1 >= 10
                if not exited:
                    peak = max(peak, favorable)
                    armed = armed or peak - entry >= 0.01 * abs(entry)
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atrs[i])
                    exited = side * close <= stop
                if exited:
                    side, entry, armed = 0, np.nan, False
                decisions[i] = side
                continue
            if np.isfinite(atrs[i]) and atrs[i] > 0:
                side = 1 if longs[i] else (-1 if shorts[i] else 0)
                entry, armed = np.nan, False
                decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        # ponytail: stops trigger next-open exits; intrabar fills require an execution-model upgrade.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=13, freq="D")
    for direction in (-1, 1):
        for expiry in (False, True):
            high = np.array([100, 100.5, 102, 104, 103, 101] + [100] * 7, dtype=float)
            low = np.array([99.5, 99.5, 101, 102, 102.5, 99] + [99.5] * 7, dtype=float)
            close = (high + low) / 2
            if expiry:
                high[:], low[:], close[:] = 100.25, 99.75, 100.0
            if direction == -1:
                high, low, close = 200 - low, 200 - high, 200 - close
            data = pd.DataFrame({"open": 100.0, "high": high, "low": low, "close": close, "volume": 1000.0}, index=index)
            ind = pd.DataFrame({"atr": 1.0, "long_entry": False, "short_entry": False}, index=index)
            ind.loc[index[0], "long_entry" if direction == 1 else "short_entry"] = True
            result = GeneratedStrategy.generate_signals(data, ind, None, GapTrapParams()).data
            end = 11 if expiry else 6
            assert result["signal"].iloc[0] == 0
            assert (result["signal"].iloc[1:end] == direction).all()
            assert (result["signal"].iloc[end:] == 0).all()
            assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
