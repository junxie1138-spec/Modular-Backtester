from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletTensionParams:
    hold_bars: int = 18
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[WaveletTensionParams]):
    strategy_id = "gen_local-w2_1791327084"

    @classmethod
    def params_type(cls) -> type[WaveletTensionParams]:
        return WaveletTensionParams

    def warmup_bars(self, params: WaveletTensionParams) -> int:
        return 43

    def indicators(self, data: pd.DataFrame, params: WaveletTensionParams) -> pd.DataFrame:
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["log_close"] = np.log(close.where(close > 0))
        out["log_return"] = out["log_close"].diff()
        out["center"] = out["log_close"].rolling(32, min_periods=32).mean()
        for half in (2, 4, 8, 16):
            out[f"recent_{half}"] = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"previous_{half}"] = out[f"recent_{half}"].shift(half)
            # Causal Haar detail, normalized by support to compare price energy across scales.
            out[f"haar_{half}"] = (
                np.sqrt(half / 2.0)
                * (out[f"recent_{half}"] - out[f"previous_{half}"])
                / half
            )
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(8, min_periods=8).mean()
        out["fine_energy"] = (out["energy_2"] + out["energy_4"]) / 2.0
        out["coarse_energy"] = (out["energy_8"] + out["energy_16"]) / 2.0
        out["fine_previous"] = out["fine_energy"].shift(4)
        out["coarse_previous"] = out["coarse_energy"].shift(4)
        out["tension"] = (
            (out["log_close"] - out["center"])
            / np.sqrt(32.0 * out["coarse_energy"].where(out["coarse_energy"] > 0))
        )
        out["realized_vol"] = out["log_return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        out["wavelet_vol"] = np.sqrt(
            (out["energy_2"] + out["energy_4"] + out["energy_8"] + out["energy_16"])
            / 4.0
        ) * np.sqrt(252.0)
        out["annual_vol"] = out[["realized_vol", "wavelet_vol"]].max(axis=1, skipna=False)
        out["risk_size"] = (params.target_vol / out["annual_vol"].clip(lower=0.01)).clip(lower=0.01, upper=0.95)
        ready = out[["tension", "fine_previous", "coarse_previous", "annual_vol"]].notna().all(axis=1)
        exhausted = (
            ready
            & (out["fine_energy"] < 0.7 * out["fine_previous"])
            & (out["coarse_energy"] > 2.0 * out["fine_energy"])
            & (out["coarse_energy"] >= 0.9 * out["coarse_previous"])
            & (out["tension"].abs() >= 0.75)
        )
        out["entry_direction"] = np.where(exhausted, -np.sign(out["tension"].fillna(0.0)), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletTensionParams,
    ) -> SignalFrame:
        if params.hold_bars < 1 or int(params.hold_bars) != params.hold_bars:
            raise ValueError("hold_bars must be a positive integer")
        candidates = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        exposure = indicators["risk_size"].replace([np.inf, -np.inf], np.nan).fillna(0.01).clip(0.01, 0.95).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        position = 0
        entry_bar = -1
        entry_size = 0.01
        for i in range(len(data)):
            if position:
                if i - entry_bar >= params.hold_bars:
                    position = 0
            elif candidates[i] in (-1, 1):
                position = int(candidates[i])
                entry_bar = i
                entry_size = float(exposure[i])
            signals[i] = position
            sizes[i] = entry_size if position else 0.01
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "entry_direction": [1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
        "risk_size": np.linspace(0.2, 0.8, 10),
    }, index=index)
    result = strategy.generate_signals(data, derived, None, WaveletTensionParams(hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, 1]
    assert np.allclose(result["size"].iloc[1:4], 0.2)
    assert np.allclose(result["size"].iloc[5:8], derived["risk_size"].iloc[4])
    assert result["size"].gt(0).all()
