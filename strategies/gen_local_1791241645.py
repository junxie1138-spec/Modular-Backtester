from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    percentile: float = 0.75
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1791241645"

    @classmethod
    def params_type(cls) -> type[SeasonalParams]:
        return SeasonalParams

    def warmup_bars(self, params: SeasonalParams) -> int:
        return 648

    def indicators(self, data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        if not 0.5 < params.percentile < 1.0:
            raise ValueError("percentile must be between 0.5 and 1")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["calendar_month"] = pd.Series(data.index.month, index=data.index)
        out["origin_month"] = out["calendar_month"].shift(18)
        denominator = data["close"].shift(18).where(lambda x: x > 0)
        out["matured_return"] = (data["close"] / denominator - 1.0).replace([np.inf, -np.inf], np.nan)
        score = pd.Series(np.nan, index=data.index, dtype=float)
        for month in range(1, 13):
            historical = out["matured_return"].where(out["origin_month"].eq(month))
            column = "seasonal_" + str(month)
            out[column] = historical.rolling(504, min_periods=21).mean()
            score = score.where(out["calendar_month"].ne(month), out[column])
        score.iloc[:521] = np.nan
        out["seasonal_score"] = score
        out["percentile"] = score.rolling(126, min_periods=126).rank(pct=True)
        out["month_start"] = out["calendar_month"].ne(out["calendar_month"].shift(1)).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalParams) -> SignalFrame:
        count = len(data)
        raw = np.zeros(count, dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        score = indicators["seasonal_score"].to_numpy(dtype=float)
        rank = indicators["percentile"].to_numpy(dtype=float)
        month_start = indicators["month_start"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        for i in range(count):
            # A previous close's order fills at this bar's open.
            if position == 0 and i > 0 and raw[i - 1] != 0:
                if np.isfinite(opening[i]) and opening[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = int(raw[i - 1])
                    entry_bar = i
                    stop = opening[i] - position * params.stop_atr * atr[i - 1]
                else:
                    continue
            if position != 0:
                stopped = not np.isfinite(close[i]) or position * (close[i] - stop) < 0
                expired = i - entry_bar >= 17
                if stopped or expired:
                    position = 0
                    raw[i] = 0
                    continue
                raw[i] = position
                continue
            eligible = month_start[i] == 1 and np.isfinite(score[i]) and np.isfinite(rank[i]) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]) and close[i] > 0
            if eligible:
                if score[i] > 0 and rank[i] >= params.percentile:
                    raw[i] = 1
                elif score[i] < 0 and rank[i] <= 1.0 - params.percentile:
                    raw[i] = -1
        df = pd.DataFrame({"signal": raw, "size": np.ones(count, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SeasonalParams()
    index = pd.bdate_range("2020-01-01", periods=25)
    prices = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"atr": 1.0, "seasonal_score": 0.02, "percentile": 0.9, "month_start": 0}, index=index)
    features.iloc[0, features.columns.get_loc("month_start")] = 1
    for direction in (1, -1):
        features["seasonal_score"] = direction * 0.02
        features["percentile"] = 0.9 if direction == 1 else 0.1
        baseline = strategy.generate_signals(prices, features, None, params).data
        assert baseline["signal"].iloc[0] == 0
        assert baseline["signal"].iloc[1:19].eq(direction).all()
        assert baseline["signal"].iloc[19:].eq(0).all()
        shocked = prices.copy()
        shocked.iloc[2, shocked.columns.get_loc("close")] = 96.0 if direction == 1 else 104.0
        stopped = strategy.generate_signals(shocked, features, None, params).data
        assert stopped["signal"].iloc[2] == direction
        assert stopped["signal"].iloc[3:].eq(0).all()
        prefix = strategy.generate_signals(shocked.iloc[:3], features.iloc[:3], None, params).data
        pd.testing.assert_frame_equal(prefix, stopped.iloc[:3])
