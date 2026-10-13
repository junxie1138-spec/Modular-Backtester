from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExhaustionParams:
    baseline_bars: int = 20
    shock_ratio: float = 1.8
    contraction_ratio: float = 0.65
    closing_pressure: float = 0.6
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ExhaustionParams]):
    strategy_id = "gen_local-w2_1791258673"

    @classmethod
    def params_type(cls):
        return ExhaustionParams

    def warmup_bars(self, params: ExhaustionParams) -> int:
        return max(2, params.baseline_bars) + 3

    def indicators(self, data: pd.DataFrame, params: ExhaustionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["baseline_atr"] = out["true_range"].rolling(
            max(2, params.baseline_bars), min_periods=max(2, params.baseline_bars)
        ).mean().shift(1)
        out["shock"] = out["true_range"].div(out["baseline_atr"].replace(0.0, np.nan))
        out["previous_shock"] = out["shock"].shift(1)
        out["contraction"] = out["true_range"].div(
            out["true_range"].shift(1).replace(0.0, np.nan)
        )
        out["return"] = data["close"].div(
            out["previous_close"].replace(0.0, np.nan)
        ) - 1.0
        out["previous_return"] = out["return"].shift(1)
        out["close_location"] = (data["close"] - data["low"]).div(
            (data["high"] - data["low"]).replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        out["direction"] = np.sign(out["return"])
        out["pressure"] = out["close_location"].where(
            out["direction"] > 0, 1.0 - out["close_location"]
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        eligible = (
            (out["previous_shock"] >= params.shock_ratio)
            & (out["contraction"] <= params.contraction_ratio)
            & (out["return"] * out["previous_return"] < 0.0)
            & (out["pressure"] >= params.closing_pressure)
        ).fillna(False)
        out["candidate"] = out["direction"].where(eligible, 0).fillna(0).astype(int)
        out["strength"] = (
            (out["previous_shock"] / max(params.shock_ratio, 1e-9) - 1.0).clip(0.0, 1.0)
            * (1.0 - out["contraction"]).clip(0.0, 1.0)
            * out["pressure"].clip(0.0, 1.0)
        ).fillna(0.0)
        out["entry_size"] = (0.25 + 0.70 * out["strength"]).clip(0.25, 0.95)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExhaustionParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(0.25).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position, entry_price, age = 0, np.nan, 0
        position_size = 0.25
        hold_limit = max(1, min(2, params.max_hold_bars))
        target = max(1e-9, params.profit_target)

        for i in range(count):
            if position:
                # The prior close's decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                age += 1
                valid_price = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                # Close-confirmed exits execute at the next open after shifting.
                if not valid_price or gain >= target or age >= hold_limit:
                    position, entry_price, age = 0, np.nan, 0
            elif candidates[i]:
                position = int(candidates[i])
                position_size = float(np.clip(entry_sizes[i], 0.25, 0.95))
                entry_price, age = np.nan, 0
            signals[i] = position
            sizes[i] = position_size if position else 0.25

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [102.0] * 7,
        "low": [98.0] * 7,
        "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    features = pd.DataFrame({
        "candidate": [1, 0, 0, -1, 0, 0, 0],
        "entry_size": [0.8, 0.25, 0.25, 0.7, 0.25, 0.25, 0.25],
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, features, None, ExhaustionParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    assert result["size"].iloc[1] == 0.8
    assert result["size"].iloc[4] == 0.7
    features["candidate"] = [-1, 0, 0, 0, 0, 0, 0]
    data.loc[index[1], "close"] = 99.0
    result = GeneratedStrategy.generate_signals(
        None, data, features, None, ExhaustionParams()
    ).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
