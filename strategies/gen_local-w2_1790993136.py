from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterEndParams:
    ma_bars: int = 40
    noise_bars: int = 10
    z_entry: float = 0.8
    z_size_cap: float = 2.5
    efficiency_limit: float = 0.45
    holding_bars: int = 18
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[QuarterEndParams]):
    strategy_id = "gen_local-w2_1790993136"

    @classmethod
    def params_type(cls) -> type[QuarterEndParams]:
        return QuarterEndParams

    def warmup_bars(self, params: QuarterEndParams) -> int:
        return int(max(2, params.ma_bars, params.noise_bars + 1))

    def indicators(self, data: pd.DataFrame, params: QuarterEndParams) -> pd.DataFrame:
        ma_bars = max(2, int(params.ma_bars))
        noise_bars = max(1, int(params.noise_bars))
        close = data["close"].astype(float)
        mean = close.rolling(ma_bars, min_periods=ma_bars).mean()
        dispersion = close.rolling(ma_bars, min_periods=ma_bars).std(ddof=0)
        z = ((close - mean) / dispersion.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        displacement = returns.rolling(noise_bars, min_periods=noise_bars).sum().abs()
        path_length = returns.abs().rolling(noise_bars, min_periods=noise_bars).sum()
        efficiency = (displacement / path_length.replace(0.0, np.nan)).clip(0.0, 1.0)
        seasonal_window = pd.Series(
            (data.index.month % 3 == 0) & (data.index.day >= 20),
            index=data.index,
            dtype=bool,
        )
        return pd.DataFrame({
            "mean": mean,
            "dispersion": dispersion,
            "z": z,
            "returns": returns,
            "displacement": displacement,
            "path_length": path_length,
            "efficiency": efficiency,
            "seasonal_window": seasonal_window,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuarterEndParams,
    ) -> SignalFrame:
        entry_threshold = max(0.01, float(params.z_entry))
        efficiency_limit = max(0.01, min(1.0, float(params.efficiency_limit)))
        eligible = (
            indicators["seasonal_window"].fillna(False)
            & indicators["z"].le(-entry_threshold)
            & indicators["efficiency"].lt(efficiency_limit)
            & indicators["z"].notna()
            & indicators["efficiency"].notna()
        ).fillna(False).to_numpy(dtype=bool)
        discount_strength = (
            -indicators["z"] / max(entry_threshold, float(params.z_size_cap))
        ).clip(0.0, 1.0).fillna(0.0)
        flow_quality = (
            1.0 - indicators["efficiency"] / efficiency_limit
        ).clip(0.0, 1.0).fillna(0.0)
        minimum = max(0.01, min(1.0, float(params.min_size)))
        maximum = max(minimum, min(1.0, float(params.max_size)))
        entry_sizes = (
            minimum + (maximum - minimum) * discount_strength * flow_quality
        ).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        holding_bars = max(1, int(params.holding_bars))
        exit_bar = -1
        active_size = 1.0
        first_entry_bar = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
                sizes[i] = active_size
            elif i >= first_entry_bar and eligible[i]:
                exit_bar = i + holding_bars
                active_size = float(entry_sizes[i])
                signal[i] = 1
                sizes[i] = active_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = QuarterEndParams(ma_bars=2, noise_bars=1, holding_bars=18)
    index = pd.date_range("2020-03-20", periods=25, freq="B")
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "seasonal_window": True,
        "z": -1.5,
        "efficiency": 0.1,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20] == 0
    assert result["size"].iloc[2:20].nunique() == 1
    assert result["size"].gt(0).all()
