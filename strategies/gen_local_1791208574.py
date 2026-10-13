from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    center_window: int = 8
    innovation_window: int = 5
    range_window: int = 20
    atr_window: int = 14
    innovation_threshold: float = 0.04
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791208574"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.center_window + params.innovation_window + 1,
            params.range_window,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        windows = (params.center_window, params.innovation_window,
                   params.range_window, params.atr_window)
        if any(w < 2 for w in windows):
            raise ValueError("Indicator windows must be at least two.")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be finite and positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")

        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["range_center"] = (data["high"] + data["low"]) / 2.0
        out["latent_center"] = out["range_center"].rolling(params.center_window).mean()
        out["center_slope"] = out["latent_center"].diff()
        out["expected_slope"] = out["center_slope"].rolling(params.innovation_window).mean().shift(1)
        out["innovation"] = (
            (out["center_slope"] - out["expected_slope"])
            / out["atr"].where(out["atr"] > 0)
        )
        out["range_low"] = data["low"].rolling(params.range_window).min()
        out["range_high"] = data["high"].rolling(params.range_window).max()
        out["range_width"] = out["range_high"] - out["range_low"]
        out["relative_position"] = (
            (data["close"] - out["range_low"])
            / out["range_width"].where(out["range_width"] > 0)
        ).clip(0.0, 1.0)
        out["mismatch_strength"] = (
            (out["innovation"].abs() / params.innovation_threshold - 1.0).clip(lower=0)
            * (2.0 * out["relative_position"] - 1.0).abs()
        )
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.15, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        innovations = indicators["innovation"].to_numpy(dtype=float)
        locations = indicators["relative_position"].to_numpy(dtype=float)
        strengths = indicators["mismatch_strength"].to_numpy(dtype=float)

        position = 0
        age = 0
        entry_price = np.nan
        frozen_atr = np.nan
        locked_size = 0.15
        for i in range(len(data)):
            if position != 0:
                # The preceding close's decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                )
                stopped = valid and (
                    (position == 1 and closes[i] < entry_price - params.stop_atr * frozen_atr)
                    or (position == -1 and closes[i] > entry_price + params.stop_atr * frozen_atr)
                )
                if not valid or stopped or age >= 2:
                    position = 0
                    age = 0
                sizes[i] = locked_size
            else:
                valid = (
                    np.isfinite(closes[i]) and closes[i] > 0
                    and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(innovations[i])
                    and np.isfinite(locations[i])
                    and np.isfinite(strengths[i])
                )
                if valid:
                    if innovations[i] > params.innovation_threshold and locations[i] < 0.45:
                        position = 1
                    elif innovations[i] < -params.innovation_threshold and locations[i] > 0.55:
                        position = -1
                    if position != 0:
                        frozen_atr = atr[i]
                        age = 0
                        strength = max(0.0, strengths[i])
                        locked_size = float(0.15 + 0.80 * strength / (1.0 + strength))
                        sizes[i] = locked_size
            signals[i] = position

        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [105.0] * 6,
        "low": [95.0] * 6,
        "close": [100.0, 97.0, 100.0, 100.0, 103.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    derived = pd.DataFrame({
        "atr": [1.0] * 6,
        "innovation": [0.2, 0.0, 0.0, -0.2, 0.0, 0.0],
        "relative_position": [0.2, 0.5, 0.5, 0.8, 0.5, 0.5],
        "mismatch_strength": [2.4, 0.0, 0.0, 2.4, 0.0, 0.0],
    }, index=index)
    stopped = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert stopped["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    data["close"] = 100.0
    derived.loc[index[3], "innovation"] = 0.0
    timed = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert timed["size"].between(0.15, 0.95).all()
    assert timed["size"].iloc[1] == timed["size"].iloc[2]
    assert timed["size"].iloc[1] > 0.15


if __name__ == "__main__":
    _self_check()
