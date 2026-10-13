from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoilParams:
    lookback: int = 60
    percentile: float = 0.85
    profit_target: float = 0.006
    max_bars: int = 2

    def __post_init__(self):
        if self.lookback < 2 or not 0.0 < self.percentile < 1.0:
            raise ValueError("Invalid percentile lookback")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if self.max_bars not in (1, 2):
            raise ValueError("max_bars must be 1 or 2")


class GeneratedStrategy(BaseStrategy[RangeRecoilParams]):
    strategy_id = "gen_local_1791237715"

    @classmethod
    def params_type(cls):
        return RangeRecoilParams

    def warmup_bars(self, params: RangeRecoilParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: RangeRecoilParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["prior_mid"] = (out["prior_high"] + out["prior_low"]) / 2.0
        out["prior_range"] = (out["prior_high"] - out["prior_low"]).where(
            out["prior_high"] > out["prior_low"]
        )
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["upper_extension"] = (
            (data["high"] - out["prior_high"]) / out["prior_range"]
        ).clip(lower=0.0)
        out["lower_extension"] = (
            (out["prior_low"] - data["low"]) / out["prior_range"]
        ).clip(lower=0.0)
        out["asymmetry"] = out["upper_extension"] - out["lower_extension"]
        out["stretch"] = out["asymmetry"].abs()
        out["expansion"] = out["range"] / out["prior_range"]
        out["relative_close"] = (data["close"] - out["prior_mid"]) / out["prior_range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        # Thresholds use only earlier completed bars.
        out["stretch_threshold"] = out["stretch"].rolling(
            params.lookback, min_periods=params.lookback
        ).quantile(params.percentile).shift(1)
        out["expansion_threshold"] = out["expansion"].rolling(
            params.lookback, min_periods=params.lookback
        ).quantile(params.percentile).shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeRecoilParams,
    ) -> SignalFrame:
        ind = indicators
        eligible = (
            (ind["stretch"] > ind["stretch_threshold"])
            & (ind["expansion"] > ind["expansion_threshold"])
            & (data["close"] > ind["prior_low"])
            & (data["close"] < ind["prior_high"])
            & (ind["asymmetry"] * ind["relative_close"] > 0.0)
        ).fillna(False)
        entries = pd.Series(
            np.where(eligible, -np.sign(ind["asymmetry"].fillna(0.0)), 0),
            index=data.index,
            dtype=int,
        ).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                # The preceding close's entry fills at this bar's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    age = 0
                age += 1
                gain = position * (closes[i] / entry_price - 1.0)
                # Close-confirmed exits fill at the following open.
                if gain >= params.profit_target or age >= params.max_bars:
                    position = 0
                    entry_price = np.nan
                    age = 0
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1.0},
        index=index,
    )
    ind = pd.DataFrame(
        {
            "stretch": [2.0, 0.0, 0.0, 0.0, 0.0],
            "stretch_threshold": 1.0,
            "expansion": 3.0,
            "expansion_threshold": 2.0,
            "prior_low": 90.0,
            "prior_high": 110.0,
            "asymmetry": -2.0,
            "relative_close": -0.1,
        },
        index=index,
    )
    params = RangeRecoilParams()
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 101.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    ind["asymmetry"] = 2.0
    ind["relative_close"] = 0.1
    data.loc[index[1], "close"] = 99.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
