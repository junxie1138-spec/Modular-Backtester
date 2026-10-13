from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_order: float = 0.6
    frac_window: int = 32
    normalize_window: int = 40
    regime_window: int = 20
    persistent_cross_rate: float = 0.15
    choppy_cross_rate: float = 0.35
    entry_z: float = 0.8
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790893349"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.frac_window + params.normalize_window + params.regime_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (
            not 0.0 < params.frac_order < 1.0
            or min(params.frac_window, params.normalize_window, params.regime_window) < 2
            or not 0.0 <= params.persistent_cross_rate < params.choppy_cross_rate <= 1.0
            or params.entry_z <= 0.0
            or params.profit_target <= 0.0
            or params.max_hold_bars < 1
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        if len(data):
            out["fractional_close"] = np.convolve(
                out["log_close"].to_numpy(), weights, mode="full"
            )[:len(data)]
            out.iloc[:params.frac_window - 1, out.columns.get_loc("fractional_close")] = np.nan
        else:
            out["fractional_close"] = pd.Series(index=data.index, dtype=float)
        # ponytail: finite fractional memory; increase frac_window for a longer tail.
        history = out["fractional_close"].shift(1).rolling(
            params.normalize_window, min_periods=params.normalize_window
        )
        out["fractional_mean"] = history.mean()
        out["fractional_std"] = history.std(ddof=0)
        out["z"] = (
            (out["fractional_close"] - out["fractional_mean"])
            / out["fractional_std"].where(out["fractional_std"] > 1e-12)
        ).replace([np.inf, -np.inf], np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["z_step"] = out["z"] - out["previous_z"]
        valid = out["z"].notna() & out["previous_z"].notna()
        out["zero_crossing"] = (
            (out["z"] >= 0.0) != (out["previous_z"] >= 0.0)
        ).astype(float).where(valid)
        out["cross_rate"] = out["zero_crossing"].rolling(
            params.regime_window, min_periods=params.regime_window
        ).mean()
        out["continuation_entry"] = (
            (out["z"] >= params.entry_z) & (out["previous_z"] < params.entry_z)
        )
        out["rebound_entry"] = (
            (out["previous_z"] < -params.entry_z)
            & (out["z"] >= -params.entry_z)
            & (out["z"] < 0.0)
        )
        out["continuation_strength"] = out["z"] / params.entry_z
        out["rebound_strength"] = out["z_step"] / params.entry_z
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        persistent = False
        active = False
        entry_bar = -1
        entry_price = np.nan
        position_size = 1.0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            rate = indicators["cross_rate"].iat[i]
            if np.isfinite(rate):
                if rate <= params.persistent_cross_rate:
                    persistent = True
                elif rate >= params.choppy_cross_rate:
                    persistent = False
            if active:
                # Read the entry bar's open only after that bar has arrived.
                if i == entry_bar:
                    entry_price = float(data["open"].iat[i])
                close = float(data["close"].iat[i])
                age = i - entry_bar + 1
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0.0
                    and np.isfinite(close)
                    and close >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or age >= params.max_hold_bars:
                    active = False
                    # Leave one flat decision; do not reenter on the exit bar.
                    continue
            elif i >= warmup and np.isfinite(rate):
                entry_key = "continuation_entry" if persistent else "rebound_entry"
                strength_key = "continuation_strength" if persistent else "rebound_strength"
                strength = indicators[strength_key].iat[i]
                if (
                    bool(indicators[entry_key].iat[i])
                    and np.isfinite(strength)
                    and np.isfinite(data["close"].iat[i])
                    and data["close"].iat[i] > 0.0
                ):
                    active = True
                    entry_bar = i + 1
                    entry_price = np.nan
                    position_size = float(0.25 + 0.75 * np.clip(strength / 3.0, 0.0, 1.0))
            if active:
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams(frac_window=3, normalize_window=3, regime_window=3, max_hold_bars=6)
    index = pd.date_range("2020-01-01", periods=28, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame(
        {
            "cross_rate": 0.1,
            "continuation_entry": False,
            "rebound_entry": False,
            "continuation_strength": 1.2,
            "rebound_strength": 0.6,
        },
        index=index,
    )
    ind.loc[index[12], "continuation_entry"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    timed = strategy.generate_signals(data, ind, ctx, params).data
    assert timed["signal"].iloc[:13].eq(0).all()
    assert timed["signal"].iloc[13:19].eq(1).all()
    assert timed["signal"].iloc[19:].eq(0).all()
    assert np.isclose(timed["size"].iat[13], 0.55)
    data.loc[index[15], "close"] = 103.0
    targeted = strategy.generate_signals(data, ind, ctx, params).data
    assert targeted["signal"].iloc[13:16].eq(1).all()
    assert targeted["signal"].iloc[16:].eq(0).all()
    ind["cross_rate"] = 0.5
    ind["continuation_entry"] = False
    ind.loc[index[12], "rebound_entry"] = True
    rebound = strategy.generate_signals(data, ind, ctx, params).data
    assert rebound["signal"].iat[13] == 1
    assert np.isclose(rebound["size"].iat[13], 0.4)
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all()
    assert flat["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
