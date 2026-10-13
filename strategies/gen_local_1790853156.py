from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MigrationParams:
    channel_bars: int = 20
    migration_bars: int = 5
    migration_ratio: float = 1.15
    atr_bars: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[MigrationParams]):
    strategy_id = "gen_local_1790853156"

    @classmethod
    def params_type(cls) -> type[MigrationParams]:
        return MigrationParams

    @staticmethod
    def warmup_bars(params: MigrationParams) -> int:
        if min(params.channel_bars, params.migration_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not np.isfinite(params.migration_ratio) or params.migration_ratio <= 1:
            raise ValueError("migration_ratio must exceed one.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive.")
        return int(max(200, params.channel_bars + 1, 2 * params.migration_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: MigrationParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["session_range"] = data["high"] - data["low"]
        out["gap"] = (data["open"] - out["previous_close"]).abs()
        out["true_range"] = pd.concat([
            out["session_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        out["channel_high"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["channel_low"] = data["low"].rolling(params.channel_bars).min().shift(1)
        out["session_vol"] = out["session_range"].rolling(params.migration_bars).mean()
        out["gap_vol"] = out["gap"].rolling(params.migration_bars).mean()
        out["previous_session_vol"] = out["session_vol"].shift(params.migration_bars)
        out["previous_gap_vol"] = out["gap_vol"].shift(params.migration_bars)
        out["migration"] = (
            (out["session_vol"] > params.migration_ratio * out["previous_session_vol"])
            & (out["gap_vol"] < out["previous_gap_vol"] / params.migration_ratio)
        )
        required = ["atr", "ma200", "channel_high", "channel_low", "session_vol", "gap_vol", "previous_session_vol", "previous_gap_vol"]
        out["ready"] = np.isfinite(out[required]).all(axis=1) & (out["atr"] > 0)
        long_setup = out["ready"] & out["migration"] & (data["close"] > out["ma200"]) & (data["close"] > out["channel_high"])
        short_setup = out["ready"] & out["migration"] & (data["close"] < out["ma200"]) & (data["close"] < out["channel_low"])
        out["setup"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MigrationParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        atrs = indicators["atr"].to_numpy(dtype=float)
        side, entry_bar, entry_atr, stop = 0, -1, np.nan, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    # Anchor once at the observed fill-bar open using decision-bar ATR.
                    stop = opens[i] - side * params.stop_atr * entry_atr
                stopped = not np.isfinite(closes[i]) or side * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                    continue
            elif setups[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                side, entry_bar, entry_atr = int(setups[i]), i + 1, atrs[i]
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    for side, closes in [(1, [100, 101, 94, 100, 100, 100]), (-1, [100, 99, 106, 100, 100, 100])]:
        data = pd.DataFrame({"open": 100.0, "close": closes}, index=index)
        ind = pd.DataFrame({"setup": [side, 0, 0, 0, 0, 0], "atr": 2.0}, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, MigrationParams()).data
        assert result["signal"].tolist() == [0, side, side, 0, 0, 0]
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    ind = pd.DataFrame({"setup": [1, 0, 0, 0, 0, 0], "atr": 2.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, MigrationParams(hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
