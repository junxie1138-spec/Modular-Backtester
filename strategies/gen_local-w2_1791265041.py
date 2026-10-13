from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathParams:
    lookback: int = 4
    area_threshold: float = 0.2
    efficiency_threshold: float = 0.4


class GeneratedStrategy(BaseStrategy[PathParams]):
    strategy_id = "gen_local-w2_1791265041"

    @classmethod
    def params_type(cls) -> type[PathParams]:
        return PathParams

    def warmup_bars(self, params: PathParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: PathParams) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0 < params.area_threshold < 1:
            raise ValueError("area_threshold must be between 0 and 1")
        if not 0 < params.efficiency_threshold < 1:
            raise ValueError("efficiency_threshold must be between 0 and 1")
        n = int(params.lookback)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = np.log(close.where(close.gt(0))).diff().replace([np.inf, -np.inf], np.nan)
        out["up"] = out["return"].clip(lower=0)
        out["down"] = (-out["return"]).clip(lower=0)
        out["up_total"] = out["up"].rolling(n, min_periods=n).sum()
        out["down_total"] = out["down"].rolling(n, min_periods=n).sum()
        out["area_numerator"] = 0.0
        # Antisymmetric level-two signature of cumulative gains and losses.
        for lag in range(1, n):
            pair = out["up"].shift(lag) * out["down"] - out["down"].shift(lag) * out["up"]
            out[f"area_lag_{lag}"] = pair.rolling(n - lag, min_periods=n - lag).sum()
            out["area_numerator"] += out[f"area_lag_{lag}"]
        out["area_denominator"] = out["up_total"] * out["down_total"]
        out["area"] = out["area_numerator"].div(out["area_denominator"].where(out["area_denominator"].gt(0))).clip(-1, 1).fillna(0.0)
        out["variation"] = out["up_total"] + out["down_total"]
        out["net_return"] = out["up_total"] - out["down_total"]
        out["efficiency"] = out["net_return"].abs().div(out["variation"].where(out["variation"].gt(0))).fillna(0.0)
        out["ready"] = out["variation"].notna() & out["area_numerator"].notna()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PathParams) -> SignalFrame:
        area = indicators["area"]
        # Efficient paths follow the later wave; inefficient paths fade it.
        direction = -np.sign(area).astype(int)
        direction = direction.where(indicators["efficiency"].ge(params.efficiency_threshold), -direction)
        candidate = direction.where(indicators["ready"] & area.abs().ge(params.area_threshold), 0)
        position = 0
        positions = np.zeros(len(data), dtype=int)
        for i, desired in enumerate(candidate.to_numpy(dtype=int)):
            if position == 0:
                position = int(desired)
            elif desired == -position:
                # Identical opposite entry condition exits; one flat bar precedes re-entry.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo() -> None:
    params = PathParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    ind = pd.DataFrame({"area": [0, -0.6, 0, 0.6, 0.6, 0, -0.6, -0.6], "efficiency": 0.8, "ready": True}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    mirror = ind.copy()
    mirror["area"] = -mirror["area"]
    reflected = GeneratedStrategy.generate_signals(None, data, mirror, None, params).data
    assert reflected["signal"].equals(-result["signal"])
    prices = pd.DataFrame({"close": 100 * np.exp(np.cumsum([0, 0.01, -0.01, 0.03, -0.02]))})
    features = GeneratedStrategy.indicators(None, prices, params)
    assert np.isclose(features["area"].iloc[-1], 0.5)
    assert not features["ready"].iloc[:4].any()
    assert GeneratedStrategy.warmup_bars(None, params) == 5


if __name__ == "__main__":
    demo()
