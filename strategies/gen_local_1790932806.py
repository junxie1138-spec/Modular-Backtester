from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    peak_window: int = 40
    signal_window: int = 8
    min_drawdown: float = 0.015
    gap_floor: float = 0.0004
    efficiency_floor: float = 0.15


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local_1790932806"

    @classmethod
    def params_type(cls) -> type[RecoilParams]:
        return RecoilParams

    def warmup_bars(self, params: RecoilParams) -> int:
        return int(params.peak_window + params.signal_window + 1)

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_peak"] = out["prior_close"].rolling(params.peak_window).max()
        out["drawdown"] = (1.0 - out["prior_close"] / out["prior_peak"].replace(0.0, np.nan)).clip(lower=0.0)
        out["gap"] = data["open"] / out["prior_close"].replace(0.0, np.nan) - 1.0
        out["loaded_gap"] = out["gap"] * out["drawdown"]
        out["depth_sum"] = out["drawdown"].rolling(params.signal_window).sum()
        out["loaded_gap_sum"] = out["loaded_gap"].rolling(params.signal_window).sum()
        out["mean_depth"] = out["depth_sum"] / params.signal_window
        out["gap_recoil"] = out["loaded_gap_sum"] / out["depth_sum"].replace(0.0, np.nan)
        out["intraday"] = data["close"] / data["open"].replace(0.0, np.nan) - 1.0
        out["intraday_net"] = out["intraday"].rolling(params.signal_window).sum()
        out["intraday_path"] = out["intraday"].abs().rolling(params.signal_window).sum()
        out["intraday_efficiency"] = out["intraday_net"] / out["intraday_path"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoilParams,
    ) -> SignalFrame:
        gap_agrees = (
            (indicators["mean_depth"] >= params.min_drawdown)
            & (indicators["gap_recoil"] > params.gap_floor)
        )
        session_agrees = indicators["intraday_efficiency"] > params.efficiency_floor
        # The same AND predicate controls entry and reversal-only exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = (gap_agrees & session_agrees).fillna(False).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RecoilParams()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=6))
    indicators = pd.DataFrame(
        {
            "mean_depth": [0.03, 0.03, 0.03, 0.03, np.nan, 0.03],
            "gap_recoil": [0.001, -0.001, 0.001, 0.001, 0.001, 0.001],
            "intraday_efficiency": [0.5, 0.5, -0.5, 0.5, 0.5, 0.5],
        },
        index=data.index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 0]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
