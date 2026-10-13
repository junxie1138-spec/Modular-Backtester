from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    coarse_share: float = 0.65
    fine_share: float = 0.10
    reversal_confirm: int = 15


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791203306"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not 0.0 < params.coarse_share < 1.0:
            raise ValueError("coarse_share must be between zero and one")
        if not 0.0 < params.fine_share < 1.0:
            raise ValueError("fine_share must be between zero and one")
        if params.reversal_confirm < 1:
            raise ValueError("reversal_confirm must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        price = out["log_close"]
        out["sum_2"] = price.rolling(2, min_periods=2).sum()
        out["sum_4"] = price.rolling(4, min_periods=4).sum()
        # Causal, orthonormally scaled Haar details over 2, 4 and 8 bars.
        out["haar_2"] = (price - price.shift(1)) / np.sqrt(2.0)
        out["haar_4"] = (out["sum_2"] - out["sum_2"].shift(2)) / 2.0
        out["haar_8"] = (out["sum_4"] - out["sum_4"].shift(4)) / np.sqrt(8.0)
        for scale in (2, 4, 8):
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2)
        out["total_energy"] = out["energy_2"] + out["energy_4"] + out["energy_8"]
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        out["coarse_fraction"] = out["energy_8"] / denominator
        out["fine_fraction"] = out["energy_2"] / denominator
        # Flat windows are valid non-entry observations, not missing data.
        out["valid"] = np.isfinite(out[["haar_2", "haar_4", "haar_8", "total_energy"]]).all(axis=1)
        out["entry_condition"] = (
            out["valid"]
            & (out["haar_8"] < 0.0)
            & (out["coarse_fraction"] >= params.coarse_share)
            & (out["fine_fraction"] <= params.fine_share)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.reversal_confirm < 1:
            raise ValueError("reversal_confirm must be positive")
        raw = np.zeros(len(data), dtype=int)
        held = False
        reversal_count = 0
        valid = indicators["valid"].fillna(False).to_numpy(dtype=bool)
        entry = indicators["entry_condition"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if not valid[i]:
                reversal_count = 0
            elif entry[i]:
                held = True
                reversal_count = 0
            elif held:
                # Exit only after the entry condition stays flipped.
                # This counts opposing evidence, not elapsed holding time.
                reversal_count += 1
                if reversal_count >= params.reversal_confirm:
                    held = False
                    reversal_count = 0
            raw[i] = int(held)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    close = np.linspace(110.0, 90.0, len(index))
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    features = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) <= 10
    assert features["haar_8"].first_valid_index() == index[7]
    assert not features["entry_condition"].iloc[:7].any()
    probe = pd.DataFrame({"valid": True, "entry_condition": False}, index=index)
    probe.loc[index[0], "entry_condition"] = True
    result = strategy.generate_signals(data, probe, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:16].eq(1).all()
    assert result["signal"].iloc[16:].eq(0).all()
    probe.loc[index[10], "entry_condition"] = True
    renewed = strategy.generate_signals(data, probe, None, params).data
    assert renewed["signal"].iloc[1:].eq(1).all()
    assert renewed["size"].gt(0).all()
