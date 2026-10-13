from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VarianceClockParams:
    clock_window: int = 3
    volatility_window: int = 20
    target_volatility: float = 0.10


class GeneratedStrategy(BaseStrategy[VarianceClockParams]):
    strategy_id = "gen_local-w2_1791278991"

    @classmethod
    def params_type(cls) -> type[VarianceClockParams]:
        return VarianceClockParams

    def warmup_bars(self, params: VarianceClockParams) -> int:
        return int(max(params.clock_window + 2, params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: VarianceClockParams) -> pd.DataFrame:
        if params.clock_window < 1 or params.volatility_window < 2:
            raise ValueError("clock_window must be positive and volatility_window at least two")
        if not np.isfinite(params.target_volatility) or params.target_volatility <= 0:
            raise ValueError("target_volatility must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"].gt(0))
        out["roc"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["clock_variance"] = out["roc"].pow(2).rolling(
            params.clock_window, min_periods=params.clock_window
        ).mean()
        out["risk_speed"] = out["roc"] / out["clock_variance"].clip(lower=1e-8)
        out["volatility"] = out["roc"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        out["acceleration"] = out["risk_speed"].diff().where(out["volatility"].notna())
        out["weight"] = (params.target_volatility / out["volatility"].clip(lower=0.01)).clip(
            lower=0.05, upper=1.0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VarianceClockParams,
    ) -> SignalFrame:
        acceleration = indicators["acceleration"]
        # Symmetric zero boundary: positive enters, negative exits; ties hold.
        decisions = pd.Series(
            np.where(acceleration.gt(0), 1.0, np.where(acceleration.lt(0), 0.0, np.nan)),
            index=data.index,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        # Exposure uses the same decision bar as the shifted position.
        df["size"] = indicators["weight"].shift(1).ffill().fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VarianceClockParams()
    index = pd.date_range("2020-01-01", periods=80)
    close = 100.0 * np.cumprod(1.0 + np.tile([0.004, 0.008, -0.003, -0.007], 20))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000000.0}, index=index
    )
    indicators = GeneratedStrategy.indicators(None, data, params)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert set(result["signal"].unique()) == {0, 1}
    assert result["signal"].iloc[:params.volatility_window + 1].eq(0).all()
    previous = indicators["acceleration"].shift(1)
    assert result.loc[previous.gt(0), "signal"].eq(1).all()
    assert result.loc[previous.lt(0), "signal"].eq(0).all()
    assert np.isfinite(result["size"]).all() and result["size"].gt(0).all()
    prefix = data.iloc[:50]
    prefix_indicators = GeneratedStrategy.indicators(None, prefix, params)
    prefix_result = GeneratedStrategy.generate_signals(None, prefix, prefix_indicators, None, params).data
    pd.testing.assert_frame_equal(result.iloc[:50], prefix_result)


if __name__ == "__main__":
    _self_check()
