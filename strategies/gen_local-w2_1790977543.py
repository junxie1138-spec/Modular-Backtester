from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    volume_capacity: float = 2.0
    stop_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local-w2_1790977543"

    @classmethod
    def params_type(cls) -> type[AbsorptionParams]:
        return AbsorptionParams

    @staticmethod
    def warmup_bars(params: AbsorptionParams) -> int:
        return 21

    @staticmethod
    def indicators(data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        if not np.isfinite(params.volume_capacity) or params.volume_capacity <= 0:
            raise ValueError("volume_capacity must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["previous_peak"] = data["close"].rolling(20, min_periods=20).max().shift(1)
        out["volume_baseline"] = data["volume"].rolling(20, min_periods=20).median().shift(1)
        out["sell_volume"] = data["volume"].where(data["close"] < data["open"], 0.0)
        out["buy_volume"] = data["volume"].where(data["close"] > data["open"], 0.0)
        out["sell_backlog"] = out["sell_volume"].rolling(3, min_periods=3).sum().shift(1)
        out["buy_backlog"] = out["buy_volume"].rolling(3, min_periods=3).sum().shift(1)
        out["drawdown"] = out["previous_peak"] - out["previous_close"]
        ready = np.isfinite(out).all(axis=1) & (out["atr"] > 0) & (out["volume_baseline"] > 0)
        drawdown = out["drawdown"] >= out["previous_atr"]
        high_volume = data["volume"] > out["volume_baseline"]
        out["long_entry"] = (
            ready & drawdown & high_volume
            & (out["sell_backlog"] >= params.volume_capacity * out["volume_baseline"])
            & (out["buy_volume"] >= 0.5 * out["sell_backlog"])
            & (data["close"] > data["open"])
            & (data["close"] > out["previous_high"])
        ).fillna(False)
        out["short_entry"] = (
            ready & drawdown & high_volume
            & (out["buy_backlog"] >= params.volume_capacity * out["volume_baseline"])
            & (out["sell_volume"] >= 0.5 * out["buy_backlog"])
            & (data["close"] < data["open"])
            & (data["close"] < out["previous_low"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AbsorptionParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        position = 0
        pending_fill = False
        entry_atr = 0.0
        stop = np.nan
        held_sessions = 0
        for i in range(len(data)):
            if position:
                if pending_fill:
                    # Capture the actual next-session open only when that session arrives.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        pending_fill = False
                        continue
                    stop = opens[i] - position * params.stop_atr * entry_atr
                    pending_fill = False
                held_sessions += 1
                invalid_close = not np.isfinite(closes[i]) or closes[i] <= 0
                stopped = (
                    (position == 1 and closes[i] < stop)
                    or (position == -1 and closes[i] > stop)
                )
                if invalid_close or stopped or held_sessions >= 2:
                    position = 0
                    held_sessions = 0
                    # The zero decision exits next session; do not reopen on this bar.
                    continue
            elif np.isfinite(atr[i]) and atr[i] > 0:
                if longs[i]:
                    position = 1
                elif shorts[i]:
                    position = -1
                if position:
                    entry_atr = atr[i]
                    pending_fill = True
                    held_sessions = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [104.0] * 7,
        "low": [96.0] * 7,
        "close": [100.0, 97.0, 100.0, 100.0, 101.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [2.0] * 7,
        "long_entry": [True, False, False, False, False, False, False],
        "short_entry": [False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, AbsorptionParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    data.loc[index[4], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, AbsorptionParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    actual = GeneratedStrategy.indicators(data, AbsorptionParams())
    assert not actual["long_entry"].any() and not actual["short_entry"].any()


if __name__ == "__main__":
    _self_check()
