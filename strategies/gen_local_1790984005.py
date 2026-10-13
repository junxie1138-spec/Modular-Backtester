from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 16
    percentile_window: int = 126
    entry_quantile: float = 0.15
    exit_quantile: float = 0.55


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790984005"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(16 + params.energy_window + params.percentile_window)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.percentile_window < 2:
            raise ValueError("Window lengths must be at least two.")
        if not 0.0 < params.entry_quantile < params.exit_quantile < 1.0:
            raise ValueError("Require 0 < entry_quantile < exit_quantile < 1.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        for half in (2, 4, 8):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = mean
            detail = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"haar_{half}"] = detail
            out[f"energy_{half}"] = detail.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).sum()
            out[f"signed_energy_{half}"] = (detail * detail.abs()).rolling(
                params.energy_window, min_periods=params.energy_window
            ).sum()
        out["total_energy"] = out[
            ["energy_2", "energy_4", "energy_8"]
        ].sum(axis=1, min_count=3)
        out["signed_energy"] = out[
            ["signed_energy_2", "signed_energy_4", "signed_energy_8"]
        ].sum(axis=1, min_count=3)
        out["pressure"] = (
            out["signed_energy"] / out["total_energy"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        history = out["pressure"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out["entry_threshold"] = history.quantile(params.entry_quantile)
        out["exit_threshold"] = history.quantile(params.exit_quantile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        # The entry state flips only when pressure reverses through its upper percentile.
        values = indicators[
            ["pressure", "entry_threshold", "exit_threshold"]
        ].to_numpy(dtype=float)
        states = np.zeros(len(data), dtype=int)
        holding = False
        for i, (pressure, entry, exit_level) in enumerate(values):
            if np.isfinite([pressure, entry, exit_level]).all():
                if holding:
                    if pressure > exit_level:
                        holding = False
                elif pressure < 0.0 and pressure <= entry:
                    holding = True
            states[i] = int(holding)
        df = pd.DataFrame({"signal": states, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {
            "pressure": [-0.8, np.nan, -0.3, 0.8, -0.8],
            "entry_threshold": [-0.7] * 5,
            "exit_threshold": [0.2] * 5,
        },
        index=index,
    )
    params = WaveletParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
    assert (result.data["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
