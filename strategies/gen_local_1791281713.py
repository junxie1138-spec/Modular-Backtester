from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    window: int = 20
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791281713"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        if params.window < 5 or not 3 <= params.hold_bars <= 5:
            raise ValueError("window must be >= 5 and hold_bars must be between 3 and 5")
        return int(params.window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        degree = pd.Series(0.0, index=data.index)
        reach = pd.Series(0.0, index=data.index)
        interior_min = pd.Series(np.inf, index=data.index)

        # Inverted-price horizontal visibility: every interior close must
        # exceed both endpoint closes. Each lag is vectorized over all bars.
        for lag in range(1, params.window):
            if lag > 1:
                interior_min = pd.Series(
                    np.minimum(interior_min.to_numpy(), close.shift(lag - 1).to_numpy()),
                    index=data.index,
                )
            previous = close.shift(lag)
            visible = (
                close.notna()
                & previous.notna()
                & (interior_min > np.maximum(close, previous))
            )
            degree = degree + visible.astype(float)
            reach = reach.where(~visible, float(lag))

        complete = close.rolling(params.window).count().eq(params.window)
        degree = degree.where(complete)
        reach = reach.where(complete)
        mean_close = close.rolling(params.window).mean()
        entry = (
            complete
            & degree.ge(3.0)
            & reach.ge(max(2, params.window // 2))
            & close.lt(mean_close)
            & close.lt(close.shift(1))
        ).fillna(False)
        return pd.DataFrame(
            {"visibility_degree": degree, "visibility_reach": reach,
             "mean_close": mean_close, "entry": entry},
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entries[i]:
                entry_bar = i
                signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VisibilityParams(window=5, hold_bars=4)
    index = pd.date_range("2020-01-01", periods=12)
    close = pd.Series([9, 12, 10, 13, 8, 9, 10, 11, 12, 13, 14, 15], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=index)
    features = GeneratedStrategy.indicators(data, params)
    assert features["visibility_degree"].iloc[4] == 3.0
    assert features["visibility_reach"].iloc[4] == 4.0
    assert features["entry"].iloc[4]
    assert features["visibility_degree"].iloc[:4].isna().all()
    forced = pd.DataFrame({"entry": True}, index=index)
    result = GeneratedStrategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
