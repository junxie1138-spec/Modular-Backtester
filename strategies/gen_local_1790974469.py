from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.55
    fractional_window: int = 32
    drift_bars: int = 4
    noise_window: int = 20
    gap_window: int = 6
    momentum_threshold: float = 0.8
    retention_threshold: float = 0.35
    profit_target: float = 0.03
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790974469"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.fractional_window + max(params.noise_window, params.drift_bars) + 1,
            params.gap_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (
            0.0 < params.fractional_order < 1.0
            and params.fractional_window >= 2
            and params.noise_window >= 2
            and params.drift_bars >= 1
            and params.gap_window >= 1
            and params.momentum_threshold > 0.0
            and 0.0 < params.retention_threshold <= 1.0
            and params.profit_target > 0.0
            and params.hold_bars >= 1
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        fractional = out["log_close"].copy()
        weight = 1.0
        # ponytail: fixed finite-memory kernel; expand the window if longer memory is needed.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * out["log_close"].shift(lag)
        out["fractional_close"] = fractional
        out["fractional_step"] = fractional.diff()
        out["fractional_scale"] = out["fractional_step"].rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).replace(0.0, np.nan)
        out["fractional_momentum"] = fractional.diff(params.drift_bars) / (
            out["fractional_scale"] * np.sqrt(params.drift_bars)
        )
        out["overnight_gap"] = (
            np.log(data["open"].where(data["open"] > 0.0)) - out["log_close"].shift(1)
        )
        out["close_displacement"] = out["log_close"].diff()
        direction = np.sign(out["overnight_gap"])
        out["retained_gap"] = direction * np.minimum(
            out["overnight_gap"].abs(),
            (direction * out["close_displacement"]).clip(lower=0.0),
        )
        out["gap_capacity"] = out["overnight_gap"].abs().rolling(
            params.gap_window, min_periods=params.gap_window
        ).sum()
        out["retained_balance"] = out["retained_gap"].rolling(
            params.gap_window, min_periods=params.gap_window
        ).sum() / out["gap_capacity"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        momentum = indicators["fractional_momentum"]
        retention = indicators["retained_balance"]
        long_entry = (
            (momentum >= params.momentum_threshold)
            & (retention >= params.retention_threshold)
        ).fillna(False)
        short_entry = (
            (momentum <= -params.momentum_threshold)
            & (retention <= -params.retention_threshold)
        ).fillna(False)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    side = 0
                else:
                    gain = side * (closes[i] / entry_price - 1.0)
                    if (
                        (np.isfinite(gain) and gain >= params.profit_target)
                        or i - entry_bar + 1 >= params.hold_bars
                    ):
                        side = 0
            elif entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                side = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params(profit_target=0.02, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame({
        "open": 100.0,
        "high": 104.0,
        "low": 96.0,
        "close": [100, 100, 103, 100, 97, 100, 100, 100, 100, 100, 100, 100],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "fractional_momentum": [2, 2, 2, -2, -2, 2, 2, 2, 2, 2, 0, 0],
        "retained_balance": [1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0.0).all()
    assert strategy.indicators(data, params)["fractional_momentum"].isna().all()


if __name__ == "__main__":
    _self_check()
