from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 8
    coarse_threshold: float = 0.65
    directional_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791278535"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return max(200, 16 + params.energy_window - 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 1:
            raise ValueError("energy_window must be positive")
        if not 0.0 < params.coarse_threshold < 1.0:
            raise ValueError("coarse_threshold must be between zero and one")
        if not 0.0 <= params.directional_threshold < 1.0:
            raise ValueError("directional_threshold must be in [0, 1)")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        log_price = np.log(data["close"].where(data["close"] > 0))
        out["log_price"] = log_price
        for half in (1, 2, 4, 8):
            recent = log_price.rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = recent
            # Causal, overlapping Haar coefficient on two adjacent blocks.
            detail = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            out[f"detail_{half}"] = detail
            out[f"energy_{half}"] = detail.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            out[f"signed_energy_{half}"] = (np.sign(detail) * detail.pow(2)).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        fine = out["energy_1"] + out["energy_2"]
        coarse = out["energy_4"] + out["energy_8"]
        out["fine_energy"] = fine
        out["coarse_energy"] = coarse
        out["coarse_share"] = coarse / (fine + coarse).where(fine + coarse > 0)
        out["coarse_bias"] = (
            out["signed_energy_4"] + out["signed_energy_8"]
        ) / coarse.where(coarse > 0)
        out["fine_bias"] = (
            out["signed_energy_1"] + out["signed_energy_2"]
        ) / fine.where(fine > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams
    ) -> SignalFrame:
        valid = indicators[["ma200", "coarse_share", "coarse_bias", "fine_bias"]].notna().all(axis=1)
        coherent = indicators["coarse_share"] >= params.coarse_threshold
        upward = indicators["coarse_bias"] > params.directional_threshold
        pullback = indicators["fine_bias"] < -params.directional_threshold
        entry = (
            valid & (data["close"] > indicators["ma200"])
            & upward & (coherent | (~coherent & pullback))
        )
        df = pd.DataFrame(index=data.index)
        # Maintain exposure while entry stays true; its reversal is the only exit.
        df["signal"] = entry.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [101.0, 101.0, 99.0, 101.0, 101.0]}, index=index)
    indicators = pd.DataFrame({
        "ma200": [100.0] * 5,
        "coarse_share": [0.8, 0.8, 0.8, 0.2, 0.2],
        "coarse_bias": [0.4, -0.4, 0.4, 0.4, 0.4],
        "fine_bias": [0.0, 0.0, 0.0, -0.4, 0.4],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1]
    prices = pd.DataFrame({"close": np.exp(np.arange(260) * 0.001)}, index=pd.date_range("2020-01-01", periods=260))
    before = strategy.indicators(prices, params)
    changed = prices.copy()
    changed.iloc[230:, 0] *= 2.0
    after = strategy.indicators(changed, params)
    assert before.iloc[:230].equals(after.iloc[:230])
    signals = strategy.generate_signals(prices, before, None, params).data
    assert signals["signal"].iloc[:200].eq(0).all()
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
