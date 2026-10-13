from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapClusterParams:
    cluster_bars: int = 10
    rank_bars: int = 63
    burden_percentile: float = 0.8
    absorption_min: float = 0.8
    min_down_gaps: int = 4
    profit_target: float = 0.045
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapClusterParams]):
    strategy_id = "gen_local-w2_1791239297"

    @classmethod
    def params_type(cls) -> type[GapClusterParams]:
        return GapClusterParams

    def warmup_bars(self, params: GapClusterParams) -> int:
        return params.cluster_bars + params.rank_bars + 2

    def indicators(self, data: pd.DataFrame, params: GapClusterParams) -> pd.DataFrame:
        if params.cluster_bars < 1 or params.rank_bars < 2:
            raise ValueError("Lookbacks must be positive and rank_bars at least two")
        if not 0.0 < params.burden_percentile <= 1.0:
            raise ValueError("burden_percentile must be in (0, 1]")
        if params.absorption_min <= 0 or not 1 <= params.min_down_gaps <= params.cluster_bars:
            raise ValueError("Invalid absorption or gap-count threshold")
        if params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("Profit target and holding period must be positive")
        previous_close = data["close"].shift(1)
        overnight = np.log(data["open"].where(data["open"] > 0) / previous_close.where(previous_close > 0))
        daytime = np.log(data["close"].where(data["close"] > 0) / data["open"].where(data["open"] > 0))
        overnight = overnight.replace([np.inf, -np.inf], np.nan)
        daytime = daytime.replace([np.inf, -np.inf], np.nan)
        downside_gap = -overnight.clip(upper=0)
        burden = downside_gap.rolling(params.cluster_bars, min_periods=params.cluster_bars).sum()
        burden_rank = burden.rolling(params.rank_bars, min_periods=params.rank_bars).rank(pct=True)
        daytime_sum = daytime.rolling(params.cluster_bars, min_periods=params.cluster_bars).sum()
        absorption = (daytime_sum / burden.where(burden > 0)).replace([np.inf, -np.inf], np.nan)
        down_gap = overnight.lt(0).astype(float).where(overnight.notna())
        down_count = down_gap.rolling(params.cluster_bars, min_periods=params.cluster_bars).sum()
        rising_close = data["close"].gt(previous_close)
        eligible = (
            burden_rank.ge(params.burden_percentile)
            & absorption.ge(params.absorption_min)
            & down_count.ge(params.min_down_gaps)
            & rising_close
        ).fillna(False)
        confirmed = eligible & eligible.shift(1, fill_value=False)
        return pd.DataFrame({
            "previous_close": previous_close,
            "overnight": overnight,
            "daytime": daytime,
            "downside_gap": downside_gap,
            "burden": burden,
            "burden_rank": burden_rank,
            "daytime_sum": daytime_sum,
            "absorption": absorption,
            "down_gap": down_gap,
            "down_count": down_count,
            "rising_close": rising_close,
            "eligible": eligible,
            "confirmed": confirmed,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapClusterParams) -> SignalFrame:
        if params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("Profit target and holding period must be positive")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        raise ValueError("Invalid entry open")
                    entry_price = opens[i]
                    entry_bar = i
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_stop = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_stop:
                    active = False
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                raw[i] = 1
            elif confirmed[i]:
                active = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 100., 120., 120., 120., 120.],
        "high": [100., 100., 120., 126., 120., 120.],
        "low": [100., 100., 120., 120., 120., 120.],
        "close": [100., 100., 120., 120., 120., 120.],
        "volume": [1000.] * 6,
    }, index=index)
    indicators = pd.DataFrame({"confirmed": [False, True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = GapClusterParams(hold_bars=3, profit_target=0.05)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    data.loc[index[3], "close"] = 126.
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.
    assert not strategy.indicators(flat, params)["confirmed"].any()
