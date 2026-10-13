from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumRecoveryParams:
    acceleration_threshold: float = 0.25
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[MomentumRecoveryParams]):
    strategy_id = "gen_local_1790923547"

    @classmethod
    def params_type(cls) -> type[MomentumRecoveryParams]:
        return MomentumRecoveryParams

    @staticmethod
    def warmup_bars(params: MomentumRecoveryParams) -> int:
        return 15

    @staticmethod
    def indicators(data: pd.DataFrame, params: MomentumRecoveryParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        close = data["close"].where(data["close"] > 0)
        roc = close.div(close.shift(3)).sub(1)
        acceleration = roc.diff()
        scale = atr.div(close).where(atr > 0)
        normalized_acceleration = acceleration.div(scale)
        long_debt = (-acceleration.clip(upper=0)).rolling(4, min_periods=4).sum().shift(1)
        short_debt = acceleration.clip(lower=0).rolling(4, min_periods=4).sum().shift(1)
        prior_roc_min = roc.rolling(4, min_periods=4).min().shift(1)
        prior_roc_max = roc.rolling(4, min_periods=4).max().shift(1)
        previous_acceleration = normalized_acceleration.shift(1)
        long_entry = (
            (roc > 0)
            & (prior_roc_min > 0)
            & (long_debt > 0)
            & (acceleration >= 0.5 * long_debt)
            & (normalized_acceleration > params.acceleration_threshold)
            & (previous_acceleration <= params.acceleration_threshold)
        )
        short_entry = (
            (roc < 0)
            & (prior_roc_max < 0)
            & (short_debt > 0)
            & (-acceleration >= 0.5 * short_debt)
            & (normalized_acceleration < -params.acceleration_threshold)
            & (previous_acceleration >= -params.acceleration_threshold)
        )
        return pd.DataFrame({
            "true_range": true_range,
            "atr": atr,
            "roc": roc,
            "acceleration": acceleration,
            "volatility_scale": scale,
            "normalized_acceleration": normalized_acceleration,
            "previous_acceleration": previous_acceleration,
            "long_debt": long_debt,
            "short_debt": short_debt,
            "prior_roc_min": prior_roc_min,
            "prior_roc_max": prior_roc_max,
            "long_entry": long_entry.fillna(False),
            "short_entry": short_entry.fillna(False),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MomentumRecoveryParams,
    ) -> SignalFrame:
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")

        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_entry = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        fixed_stop = np.nan

        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    # The preceding close's decision fills at this bar's open.
                    if np.isfinite(opens[i]) and opens[i] > 0:
                        fixed_stop = opens[i] - position * params.stop_atr * atr[i - 1]
                    else:
                        position = 0
                if position != 0:
                    stopped = np.isfinite(closes[i]) and (
                        (position == 1 and closes[i] < fixed_stop)
                        or (position == -1 and closes[i] > fixed_stop)
                    )
                    if stopped or i - entry_bar + 1 >= 4:
                        position = 0
            elif (
                i >= GeneratedStrategy.warmup_bars(params) - 1
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(closes[i]) and closes[i] > 0
            ):
                if long_entry[i]:
                    position = 1
                elif short_entry[i]:
                    position = -1
                if position != 0:
                    entry_bar = i + 1
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=22, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "atr": 1.0, "long_entry": False, "short_entry": False,
    }, index=index)
    derived.loc[index[14], "long_entry"] = True
    params = MomentumRecoveryParams(stop_atr=2.0)
    data.loc[index[16], "close"] = 97.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[15:17] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    data.loc[index[16], "close"] = 100.0
    derived.loc[index[14], "long_entry"] = False
    derived.loc[index[14], "short_entry"] = True
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    expected[:] = 0
    expected[15:19] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    data.loc[index[16], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    expected[:] = 0
    expected[15:17] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)


if __name__ == "__main__":
    _self_check()
