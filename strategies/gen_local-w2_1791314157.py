from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 4
    percentile_window: int = 126
    entry_quantile: float = 0.75


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791314157"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(params.path_bars + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.path_bars < 2 or params.percentile_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.5 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between 0.5 and 1.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["return_increment"] = out["log_close"].diff()
        out["relative_range"] = (data["high"] - data["low"]) / close
        out["log_range"] = np.log(out["relative_range"].where(out["relative_range"] > 0))
        out["range_increment"] = out["log_range"].diff()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)

        x = out["return_increment"]
        y = out["range_increment"]
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature: earlier increments precede later ones.
        for earlier in range(params.path_bars - 1, -1, -1):
            for later in range(earlier - 1, -1, -1):
                area += 0.5 * (
                    x.shift(earlier) * y.shift(later)
                    - y.shift(earlier) * x.shift(later)
                )
        out["signature_area"] = area
        out["path_return"] = x.rolling(
            params.path_bars, min_periods=params.path_bars
        ).sum()
        out["area_threshold"] = area.rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        ready = indicators[
            ["signature_area", "area_threshold", "path_return"]
        ].notna().all(axis=1)
        entry_condition = (
            ready
            & (indicators["signature_area"] > indicators["area_threshold"])
            & (indicators["path_return"] < 0)
        )
        df = pd.DataFrame(index=data.index)
        # Remain long while qualified; exit only when this same condition flips false.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame(
        {
            "signature_area": [2.0, 0.0, 2.0, np.nan],
            "area_threshold": [1.0, 1.0, 1.0, 1.0],
            "path_return": [-0.01, -0.01, -0.01, -0.01],
        },
        index=index,
    )
    result = GeneratedStrategy().generate_signals(
        data, derived, None, SignatureParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 0, 1]
    assert result["size"].gt(0).all()
