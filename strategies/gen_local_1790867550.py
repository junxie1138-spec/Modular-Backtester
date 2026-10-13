from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletImpulseParams:
    energy_window: int = 20
    burst_threshold: float = 1.5
    fine_share_threshold: float = 0.55


class GeneratedStrategy(BaseStrategy[WaveletImpulseParams]):
    strategy_id = "gen_local_1790867550"

    @classmethod
    def params_type(cls) -> type[WaveletImpulseParams]:
        return WaveletImpulseParams

    def warmup_bars(self, params: WaveletImpulseParams) -> int:
        return int(max(params.energy_window + 5, 9))

    def indicators(self, data: pd.DataFrame, params: WaveletImpulseParams) -> pd.DataFrame:
        if params.energy_window < 2:
            raise ValueError("energy_window must be at least 2")
        if not np.isfinite(params.burst_threshold) or params.burst_threshold <= 0:
            raise ValueError("burst_threshold must be finite and positive")
        if not 0 < params.fine_share_threshold < 1:
            raise ValueError("fine_share_threshold must be between zero and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_return"] = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        r = out["log_return"]
        # Causal Haar details compare adjacent return blocks at three scales.
        for width in (1, 2, 4):
            block = r.rolling(width, min_periods=width).sum()
            out[f"detail_{width}"] = (block - block.shift(width)) / np.sqrt(2.0 * width)
            out[f"energy_{width}"] = out[f"detail_{width}"].pow(2)

        out["fast_energy"] = out["energy_1"] + out["energy_2"]
        out["total_energy"] = out["fast_energy"] + out["energy_4"]
        out["fine_share"] = out["fast_energy"] / out["total_energy"].where(out["total_energy"] > 0)
        out["baseline_energy"] = out["fast_energy"].rolling(
            params.energy_window, min_periods=params.energy_window
        ).mean().shift(1)
        out["burst"] = out["fast_energy"] / out["baseline_energy"].where(out["baseline_energy"] > 0)
        out["momentum_2"] = r.rolling(2, min_periods=2).sum()
        out = out.replace([np.inf, -np.inf], np.nan)

        qualified = (
            out["burst"].ge(params.burst_threshold)
            & out["fine_share"].ge(params.fine_share_threshold)
        )
        positive = (
            out["detail_1"].gt(0) & out["detail_2"].gt(0)
            & out["momentum_2"].gt(0) & r.gt(0)
        )
        negative = (
            out["detail_1"].lt(0) & out["detail_2"].lt(0)
            & out["momentum_2"].lt(0) & r.lt(0)
        )
        out["entry_direction"] = np.select(
            [qualified & positive, qualified & negative], [1, -1], default=0
        ).astype(int)
        out["strength"] = (
            out["fine_share"] * out["burst"] / (out["burst"] + params.burst_threshold)
        ).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletImpulseParams,
    ) -> SignalFrame:
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        position = 0
        entry_size = 0.2
        for i, direction in enumerate(entries):
            if position != 0 and direction == -position:
                # An opposite qualification exits; reversal entry needs a later bar.
                position = 0
                entry_size = 0.2
            elif position == 0 and direction != 0:
                position = int(direction)
                entry_size = float(0.2 + 0.75 * strengths[i])
            signals[i] = position
            sizes[i] = entry_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"entry_direction": [1, 0, -1, -1, 0, 1],
         "strength": [0.8, 0.2, 0.9, 0.7, 0.2, 0.6]},
        index=index,
    )
    result = GeneratedStrategy().generate_signals(
        data, indicators, None, WaveletImpulseParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert np.allclose(result["size"], [0.2, 0.8, 0.8, 0.2, 0.725, 0.725])
    prices = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=50, freq="D"),
    )
    flat = GeneratedStrategy().indicators(prices, WaveletImpulseParams())
    assert flat["entry_direction"].eq(0).all()
    assert flat["strength"].eq(0).all()


if __name__ == "__main__":
    _self_check()
