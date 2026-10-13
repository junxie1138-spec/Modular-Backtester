from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    history_bars: int = 504
    momentum_bars: int = 63
    min_samples: int = 18
    seasonal_quantile: float = 0.75
    momentum_quantile: float = 0.55
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1791200221"

    @classmethod
    def params_type(cls) -> type[SeasonalParams]:
        return SeasonalParams

    def warmup_bars(self, params: SeasonalParams) -> int:
        return int(max(params.history_bars + params.max_hold_bars,
                       params.momentum_bars + 1))

    def indicators(self, data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result["daily_return"] = data["close"].pct_change(fill_method=None)
        result["cohort_return"] = data["close"].pct_change(
            periods=params.max_hold_bars, fill_method=None
        )
        result["cohort_month"] = pd.Series(
            data.index.month, index=data.index, dtype=float
        ).shift(params.max_hold_bars)
        scores = []
        for month in range(1, 13):
            valid = (
                result["cohort_month"].eq(month)
                & result["cohort_return"].notna()
                & np.isfinite(result["cohort_return"])
            )
            count = valid.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            total = result["cohort_return"].where(valid, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            column = f"month_score_{month}"
            result[column] = (total / count.replace(0.0, np.nan)).where(
                count >= params.min_samples
            )
            scores.append(column)
        score_matrix = result[scores]
        result["seasonal_threshold"] = score_matrix.quantile(
            params.seasonal_quantile, axis=1
        ).where(score_matrix.notna().all(axis=1))
        result["seasonal_score"] = np.nan
        for month, column in enumerate(scores, start=1):
            mask = data.index.month == month
            result.loc[mask, "seasonal_score"] = result.loc[mask, column]
        result["momentum_threshold"] = result["daily_return"].shift(1).rolling(
            params.momentum_bars, min_periods=params.momentum_bars
        ).quantile(params.momentum_quantile)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalParams,
    ) -> SignalFrame:
        required = ["seasonal_score", "seasonal_threshold", "daily_return",
                    "momentum_threshold"]
        eligible = (
            indicators[required].notna().all(axis=1)
            & (indicators["seasonal_score"] > indicators["seasonal_threshold"])
            & (indicators["seasonal_score"] > 0.0)
            & (indicators["daily_return"] > indicators["momentum_threshold"])
            & (indicators["daily_return"] > 0.0)
        ).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        month_keys = data.index.year.to_numpy() * 12 + data.index.month.to_numpy()
        raw = np.zeros(len(data), dtype=int)
        holding = False
        pending_entry = False
        entry_price = 0.0
        entry_bar = 0
        last_entry_month = -1
        for i in range(len(data)):
            # Yesterday's decision fills at today's open; no future open is read.
            if pending_entry:
                pending_entry = False
                holding = bool(np.isfinite(opens[i]) and opens[i] > 0.0)
                if holding:
                    entry_price = opens[i]
                    entry_bar = i
            if holding:
                target_hit = bool(
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_hit = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_hit:
                    holding = False
                else:
                    raw[i] = 1
            elif eligible[i] and month_keys[i] != last_entry_month:
                raw[i] = 1
                pending_entry = True
                last_entry_month = int(month_keys[i])
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [106.0] * 8,
        "low": [99.0] * 8,
        "close": [100.0, 101.0, 105.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        "seasonal_score": 0.03,
        "seasonal_threshold": 0.02,
        "daily_return": 0.01,
        "momentum_threshold": 0.005,
    }, index=index)
    params = SeasonalParams(max_hold_bars=3)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
