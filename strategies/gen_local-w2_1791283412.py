from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ContagionParams:
    lookback: int = 12
    entry_margin: float = 0.04
    sizing_gain: float = 3.0
    min_size: float = 0.15
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ContagionParams]):
    strategy_id = "gen_local-w2_1791283412"

    @classmethod
    def params_type(cls) -> type[ContagionParams]:
        return ContagionParams

    def warmup_bars(self, params: ContagionParams) -> int:
        return int(params.lookback) + 2

    def indicators(self, data: pd.DataFrame, params: ContagionParams) -> pd.DataFrame:
        if params.lookback < 2 or not 0.0 <= params.entry_margin < 1.0:
            raise ValueError("Invalid lookback or entry_margin")
        if not np.isfinite(params.sizing_gain) or params.sizing_gain <= 0:
            raise ValueError("sizing_gain must be finite and positive")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Require 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        out["returns"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["positive_return"] = out["returns"].clip(lower=0.0)
        out["negative_return"] = (-out["returns"]).clip(lower=0.0)
        out["seed"] = out["positive_return"].shift(1)
        out["susceptible_day"] = out["returns"].le(0.0).astype(float).where(out["returns"].notna())
        out["transmission_pair"] = out["seed"] * out["positive_return"]
        out["recovery_pair"] = out["seed"] * out["negative_return"]
        window = int(params.lookback)
        out["transmission"] = out["transmission_pair"].rolling(window, min_periods=window).sum()
        out["recovery"] = out["recovery_pair"].rolling(window, min_periods=window).sum()
        out["susceptible"] = out["susceptible_day"].rolling(window, min_periods=window).mean()
        out["drift"] = out["returns"].rolling(window, min_periods=window).sum()
        out["pair_energy"] = out["transmission"] + out["recovery"]
        out["margin"] = (out["susceptible"] * out["transmission"] - out["recovery"]) / out["pair_energy"].replace(0.0, np.nan)
        out["ready"] = out[["margin", "drift"]].notna().all(axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ContagionParams) -> SignalFrame:
        bullish = indicators["ready"] & indicators["margin"].gt(params.entry_margin) & indicators["drift"].gt(0.0)
        df = pd.DataFrame(index=data.index)
        # A false entry condition reverses the position to flat; there are no other exits.
        df["signal"] = bullish.astype(int)
        strength = ((indicators["margin"] - params.entry_margin) * params.sizing_gain).clip(0.0, 1.0)
        df["size"] = params.min_size + (params.max_size - params.min_size) * strength
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ContagionParams()
    strategy = GeneratedStrategy()
    returns = np.array([0.012, 0.009, -0.002] * 12 + [-0.015] * 20)
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({name: close for name in ("open", "high", "low", "close")}, index=pd.date_range("2020-01-01", periods=len(close)))
    data["volume"] = 1000000.0
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = (indicators["ready"] & indicators["margin"].gt(params.entry_margin) & indicators["drift"].gt(0.0)).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].max() == 1 and result["signal"].iloc[-1] == 0
    assert result["size"].between(params.min_size, params.max_size).all()
    assert result.loc[result["signal"].eq(1), "size"].nunique() > 1
    prefix = data.iloc[:30]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(result.iloc[:30], prefix_result)
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    flat_result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert flat_result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
