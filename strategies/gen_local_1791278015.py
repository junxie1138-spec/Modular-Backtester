from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapConfirmationParams:
    trend_lookback: int = 20
    atr_lookback: int = 14
    min_gap_atr: float = 0.35
    min_efficiency: float = 0.12
    confirmation_fraction: float = 0.65
    stop_atr: float = 1.3
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GapConfirmationParams]):
    strategy_id = "gen_local_1791278015"

    @classmethod
    def params_type(cls) -> type[GapConfirmationParams]:
        return GapConfirmationParams

    @staticmethod
    def warmup_bars(params: GapConfirmationParams) -> int:
        return int(max(params.trend_lookback, params.atr_lookback) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapConfirmationParams) -> pd.DataFrame:
        if params.trend_lookback < 1 or params.atr_lookback < 1:
            raise ValueError("Lookbacks must be positive")
        if params.min_gap_atr <= 0 or params.min_efficiency <= 0:
            raise ValueError("Signal thresholds must be positive")
        if not 0.5 <= params.confirmation_fraction <= 1.0:
            raise ValueError("confirmation_fraction must be between 0.5 and 1")
        if params.stop_atr <= 0 or not 3 <= params.hold_days <= 5:
            raise ValueError("Use a positive stop and a three-to-five-session hold")

        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["previous_close"] = previous_close
        out["close_change"] = data["close"].diff()
        out["gap"] = data["open"] - previous_close
        out["gap_direction"] = np.sign(out["gap"])
        out["range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_lookback).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["trend_move"] = out["close_change"].rolling(params.trend_lookback).sum()
        out["trend_noise"] = out["true_range"].rolling(params.trend_lookback).sum()
        out["efficiency"] = out["trend_move"] / out["trend_noise"].replace(0.0, np.nan)
        out["close_location"] = (
            (data["close"] - data["low"]) / out["range"].replace(0.0, np.nan)
        )
        out["directional_close_location"] = out["close_location"].where(
            out["gap_direction"] > 0, 1.0 - out["close_location"]
        )
        out["body"] = data["close"] - data["open"]
        first_confirmation = (
            (out["prior_atr"] > 0)
            & (out["gap"].abs() >= params.min_gap_atr * out["prior_atr"])
            & (out["gap_direction"] * out["efficiency"] >= params.min_efficiency)
            & (out["gap_direction"] * out["body"] > 0)
            & (out["directional_close_location"] >= params.confirmation_fraction)
        ).fillna(False)
        out["first_side"] = out["gap_direction"].where(first_confirmation, 0).fillna(0).astype(int)
        out["confirmation_side"] = out["first_side"].shift(1).fillna(0).astype(int)
        out["second_close_location"] = out["close_location"].where(
            out["confirmation_side"] > 0, 1.0 - out["close_location"]
        )
        second_confirmation = (
            (out["confirmation_side"] != 0)
            & (out["confirmation_side"] * out["close_change"] > 0)
            & (out["second_close_location"] >= params.confirmation_fraction)
            & (out["atr"] > 0)
        ).fillna(False)
        out["entry_side"] = out["confirmation_side"].where(second_confirmation, 0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapConfirmationParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_side"].fillna(0).to_numpy(dtype=int)
        targets = np.zeros(len(data), dtype=int)
        side = 0
        pending = 0
        entry_bar = -1
        stop_price = np.nan

        for i in range(len(data)):
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    side = pending
                    entry_bar = i
                    stop_price = opens[i] - side * params.stop_atr * atr[i - 1]
                pending = 0

            if side:
                stopped = np.isfinite(closes[i]) and side * (closes[i] - stop_price) <= 0
                expired = i - entry_bar + 1 >= params.hold_days
                if stopped or expired:
                    side = 0
                targets[i] = side
                continue

            if entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(entries[i])
                targets[i] = pending

        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapConfirmationParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 97.0,
        "close": [100, 100, 98, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "atr": 1.0,
        "entry_side": [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    derived["entry_side"] = [1] + [0] * 11
    data["close"] = 100.0
    data.loc[index[2], "open"] = 110.0
    data.loc[index[2], "close"] = 109.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    derived["entry_side"] = [-1] + [0] * 11
    data["open"] = 100.0
    data["close"] = 100.0
    data.loc[index[2], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    calculated = GeneratedStrategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert calculated["entry_side"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.trend_lookback + 2


if __name__ == "__main__":
    _self_check()
