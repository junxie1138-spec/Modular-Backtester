from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterReliefParams:
    atr_window: int = 5
    percentile_window: int = 126
    volatility_quantile: float = 0.75
    profit_target: float = 0.02
    max_hold_bars: int = 8

    def __post_init__(self):
        if self.atr_window < 2 or self.percentile_window < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < self.volatility_quantile < 1.0:
            raise ValueError("Quantile must be strictly between zero and one.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("Profit target must be finite and positive.")
        if self.max_hold_bars < 1:
            raise ValueError("Holding period must be positive.")


class GeneratedStrategy(BaseStrategy[QuarterReliefParams]):
    strategy_id = "gen_local_1790883580"

    @classmethod
    def params_type(cls):
        return QuarterReliefParams

    def warmup_bars(self, params: QuarterReliefParams) -> int:
        return int(params.atr_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: QuarterReliefParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["atr_fraction"] = (
            out["atr"] / data["close"].where(data["close"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["volatility_threshold"] = out["atr_fraction"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.volatility_quantile)
        out["quarter_end_week"] = (
            data.index.month.isin([3, 6, 9, 12]) & (data.index.day >= 24)
        )
        out["cooling"] = out["atr_fraction"] < out["atr_fraction"].shift(1)
        out["entry"] = (
            out["quarter_end_week"]
            & out["cooling"]
            & (out["atr_fraction"] > out["volatility_threshold"])
            & out["volatility_threshold"].notna()
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuarterReliefParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        last_quarter = -1
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    active = False
                    continue
                signal[i] = 1
            elif entries[i]:
                quarter = data.index[i].year * 4 + data.index[i].quarter
                if quarter != last_quarter:
                    active = True
                    entry_bar = i + 1
                    last_quarter = quarter
                    signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = QuarterReliefParams(max_hold_bars=3)
    index = pd.bdate_range("2020-03-24", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    entries = pd.DataFrame({"entry": [True] * 6}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[1], "open"] = 101.0
    data.loc[index[1], "close"] = 102.0
    data.loc[index[2], "close"] = 104.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    derived = strategy.indicators(data, params)
    assert not derived["entry"].any()


if __name__ == "__main__":
    _self_check()
