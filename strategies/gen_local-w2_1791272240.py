from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalPressureParams:
    frac_window: int = 16
    frac_order: float = 0.65
    pressure_window: int = 4
    reversal_threshold: float = 0.35
    vol_window: int = 20
    target_vol: float = 0.12
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[FractionalPressureParams]):
    strategy_id = "gen_local-w2_1791272240"

    @classmethod
    def params_type(cls) -> type[FractionalPressureParams]:
        return FractionalPressureParams

    @staticmethod
    def warmup_bars(params: FractionalPressureParams) -> int:
        return int(max(params.frac_window + params.pressure_window, params.vol_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: FractionalPressureParams) -> pd.DataFrame:
        if not (params.frac_window >= 2 and params.pressure_window >= 1
                and params.vol_window >= 2 and 0 < params.frac_order < 1
                and 0 < params.reversal_threshold < 1
                and params.target_vol > 0 and 0 < params.max_size <= 1):
            raise ValueError("Invalid fractional-pressure parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        # ponytail: fixed finite fractional memory; extend frac_window if truncation matters.
        out["fractional_close"] = sum(
            weights[k] * close.shift(k) for k in range(params.frac_window)
        )
        out["innovation"] = out["fractional_close"].diff()
        out["net_pressure"] = out["innovation"].rolling(params.pressure_window).sum()
        out["gross_pressure"] = out["innovation"].abs().rolling(params.pressure_window).sum()
        out["pressure"] = (out["net_pressure"] / out["gross_pressure"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["realized_vol"] = out["return"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalPressureParams,
    ) -> SignalFrame:
        pressure = indicators["pressure"]
        ready = pressure.notna() & indicators["realized_vol"].notna()
        state = pd.Series(np.nan, index=data.index, dtype=float)
        state.loc[ready & (pressure >= params.reversal_threshold)] = 1.0
        state.loc[ready & (pressure <= -params.reversal_threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = state.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        exposure = (params.target_vol / indicators["realized_vol"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).clip(lower=min(0.05, params.max_size), upper=params.max_size).fillna(
            min(0.05, params.max_size)
        )
        df["size"] = exposure.shift(1).fillna(min(0.05, params.max_size)).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = FractionalPressureParams()
    idx = pd.date_range("2020-01-01", periods=6)
    bars = pd.DataFrame({"close": np.arange(100.0, 106.0)}, index=idx)
    features = pd.DataFrame(
        {"pressure": [np.nan, 0.6, 0.0, -0.6, 0.0, 0.6],
         "realized_vol": [np.nan, 0.2, 0.4, 0.2, 0.2, 0.2]}, index=idx
    )
    result = GeneratedStrategy.generate_signals(bars, features, None, p).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[2], 0.6)
    assert np.isclose(result["size"].iloc[3], 0.3)
    assert result["size"].gt(0).all()
