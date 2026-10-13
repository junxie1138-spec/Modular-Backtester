from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 4
    reversal_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791337670"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        if params.path_bars < 2:
            raise ValueError("path_bars must be at least 2")
        if not np.isfinite(params.reversal_threshold) or params.reversal_threshold <= 0:
            raise ValueError("reversal_threshold must be finite and positive")
        return int(2 * params.path_bars)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        n = params.path_bars
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["intraday_return"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        r = out["intraday_return"]
        out["absolute_flow"] = r.abs().rolling(n, min_periods=n).sum()
        # Exact antisymmetric level-two area for increments (1/n, r).
        out["weighted_flow"] = sum((n - 1 - 2 * lag) * r.shift(lag) for lag in range(n))
        denominator = 2.0 * n * out["absolute_flow"]
        area = out["weighted_flow"] / denominator.where(denominator > 0)
        out["signature_area"] = area.mask(out["absolute_flow"].eq(0), 0.0)
        out["previous_area"] = out["signature_area"].shift(n)
        out["area_change"] = out["signature_area"] - out["previous_area"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        change = indicators["area_change"]
        state = pd.Series(np.nan, index=data.index, dtype=float)
        # Symmetric reversal boundaries; neutral observations retain exposure.
        state.loc[change > params.reversal_threshold] = 1.0
        state.loc[change < -params.reversal_threshold] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = state.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=6))
    indicators = pd.DataFrame({"area_change": [np.nan, 0.2, 0.0, -0.2, 0.0, 0.2]}, index=data.index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SignatureParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
