from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureGapParams:
    percentile_window: int = 126
    entry_percentile: float = 0.10
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureGapParams]):
    strategy_id = "gen_local_1790929396"

    @classmethod
    def params_type(cls) -> type[SignatureGapParams]:
        return SignatureGapParams

    def warmup_bars(self, params: SignatureGapParams) -> int:
        return int(params.percentile_window) + 3

    def indicators(self, data: pd.DataFrame, params: SignatureGapParams) -> pd.DataFrame:
        if params.percentile_window < 2:
            raise ValueError("percentile_window must be at least 2")
        if not 0.0 < params.entry_percentile < 0.5:
            raise ValueError("entry_percentile must be between 0 and 0.5")
        previous_close = data["close"].shift(1).where(data["close"].shift(1) > 0)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        gap = np.log(opening / previous_close)
        intraday = np.log(closing / opening)
        prior_gap = gap.shift(1)
        prior_intraday = intraday.shift(1)
        # Half the antisymmetric second-order signature of the four axis-aligned increments.
        area = 0.5 * (
            prior_gap * prior_intraday
            + prior_gap * intraday
            - prior_intraday * gap
            + gap * intraday
        )
        threshold = area.rolling(
            int(params.percentile_window), min_periods=int(params.percentile_window)
        ).quantile(params.entry_percentile).shift(1)
        return pd.DataFrame(
            {"gap": gap, "intraday": intraday, "area": area, "threshold": threshold},
            index=data.index,
        ).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureGapParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        entries = (
            indicators.notna().all(axis=1)
            & (indicators["gap"] < 0)
            & (indicators["intraday"] > 0)
            & (indicators["area"] < indicators["threshold"])
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, enter in enumerate(entries):
            if remaining > 0:
                remaining -= 1
                signal[i] = int(remaining > 0)
            elif enter:
                remaining = int(params.hold_bars)
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"gap": -0.01, "intraday": 0.01, "area": -0.001, "threshold": -0.0001},
        index=index,
    )
    strategy = GeneratedStrategy()
    for holding, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1, 0]),
        (2, [0, 1, 1, 0, 1, 1, 0, 1, 1]),
    ):
        result = strategy.generate_signals(
            data, indicators, None, SignatureGapParams(hold_bars=holding)
        ).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
    indicators.loc[:, "threshold"] = np.nan
    assert strategy.generate_signals(
        data, indicators, None, SignatureGapParams()
    ).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
