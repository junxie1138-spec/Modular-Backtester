from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 12
    entropy_enter: float = 0.72
    entropy_reset: float = 0.90
    middle_share_min: float = 0.45
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791324031"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return max(200, 8 + max(2, int(params.energy_window)) + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        returns = np.log(close.where(close > 0)).diff()
        window = max(2, int(params.energy_window))
        energies = pd.DataFrame(index=data.index)
        for scale in (2, 4, 8):
            half = scale // 2
            mean = returns.rolling(half, min_periods=half).mean()
            detail = np.sqrt(scale / 2.0) * (mean - mean.shift(half))
            out[f"detail_{scale}"] = detail
            energies[f"energy_{scale}"] = detail.pow(2).rolling(window, min_periods=window).mean()
        out = out.join(energies)
        total = energies.sum(axis=1, min_count=3)
        shares = energies.div(total.where(total > 0), axis=0)
        out["fine_share"] = shares["energy_2"]
        out["middle_share"] = shares["energy_4"]
        entropy_terms = shares * np.log(shares.clip(lower=1e-12))
        out["entropy"] = -entropy_terms.sum(axis=1, min_count=3) / np.log(3.0)
        out["ma_200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        if not 0.0 <= params.entropy_enter < params.entropy_reset <= 1.0:
            raise ValueError("Entropy thresholds must satisfy 0 <= enter < reset <= 1")
        if not 0.0 < params.middle_share_min <= 1.0:
            raise ValueError("middle_share_min must be in (0, 1]")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        entropy = indicators["entropy"].to_numpy(dtype=float)
        prior_fine = indicators["fine_share"].shift(1)
        eligible = (
            (data["close"] > indicators["ma_200"])
            & (indicators["middle_share"] >= params.middle_share_min)
            & (prior_fine >= params.middle_share_min)
            & (indicators["fine_share"] < prior_fine)
            & (indicators["detail_4"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        organized = False
        armed = False
        exit_bar = -1
        for i in range(len(data)):
            transition = False
            if np.isfinite(entropy[i]):
                if entropy[i] >= params.entropy_reset:
                    organized = False
                    armed = True
                elif entropy[i] <= params.entropy_enter and not organized:
                    organized = True
                    transition = armed
                    armed = False
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    exit_bar = -1
                continue
            if transition and eligible[i]:
                signals[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"close": 101.0}, index=index)
    indicators = pd.DataFrame({
        "entropy": [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        "fine_share": [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1],
        "middle_share": 0.7,
        "detail_4": 0.01,
        "ma_200": 100.0,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, WaveletParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
