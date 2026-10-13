from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletRotationParams:
    threshold_window: int = 63
    entry_percentile: float = 0.85
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletRotationParams]):
    strategy_id = "gen_local_1791333912"

    @classmethod
    def params_type(cls):
        return WaveletRotationParams

    @staticmethod
    def warmup_bars(params: WaveletRotationParams) -> int:
        if params.threshold_window < 2 or not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("Invalid percentile window or probability")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two")
        return int(params.threshold_window + 10)

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletRotationParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        price = data["close"].astype(float)
        out["log_price"] = np.log(price.where(np.isfinite(price) & (price > 0.0)))
        for half in (1, 2, 4):
            block = out["log_price"].rolling(half, min_periods=half).sum()
            out[f"block_{half}"] = block
            out[f"detail_{half}"] = (block - block.shift(half)) / np.sqrt(2.0 * half)
            out[f"energy_{half}"] = out[f"detail_{half}"].pow(2)
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["fine_share"] = out["energy_1"] / out["total_energy"].where(out["total_energy"] > 0.0)
        out["rotation"] = out["fine_share"].diff()
        out["rotation_threshold"] = out["rotation"].shift(1).rolling(
            params.threshold_window, min_periods=params.threshold_window
        ).quantile(params.entry_percentile)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            (out["rotation"] > out["rotation_threshold"])
            & (out["rotation"] > 0.0)
            & (out["energy_4"].shift(1) > out["energy_1"].shift(1))
            & (out["energy_1"] > out["energy_1"].shift(1))
            & (out["detail_1"] > 0.0)
            & (out["detail_2"] > 0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletRotationParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        setup = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The entry decision was yesterday; today's open is now known.
                    entry_price = opens[i]
                held_bars = i - entry_bar + 1
                invalid = not np.isfinite(entry_price) or entry_price <= 0.0
                target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                # Close-confirmed exits execute at the following open.
                if invalid or target or held_bars >= params.max_hold_bars:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif setup[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                signal[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [102.0] * 5,
        "low": [99.0] * 5,
        "close": [100.0] * 5,
        "volume": [1000.0] * 5,
    }, index=index)
    ind = pd.DataFrame({"setup": [True, False, False, False, False]}, index=index)
    params = WaveletRotationParams()
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data.iloc[1, data.columns.get_loc("close")] = 101.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    assert (result["size"] > 0.0).all()
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["setup"].any()


if __name__ == "__main__":
    _self_check()
