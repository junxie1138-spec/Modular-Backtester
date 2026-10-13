from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticStreakParams:
    channel_bars: int = 20
    percentile_bars: int = 126
    streak_percentile: float = 0.85
    atr_bars: int = 14
    elastic_atr: float = 0.8
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ElasticStreakParams]):
    strategy_id = "gen_local_1791254320"

    @classmethod
    def params_type(cls) -> type[ElasticStreakParams]:
        return ElasticStreakParams

    def warmup_bars(self, params: ElasticStreakParams) -> int:
        return int(params.channel_bars + params.percentile_bars + params.atr_bars + 3)

    def indicators(self, data: pd.DataFrame, params: ElasticStreakParams) -> pd.DataFrame:
        if min(params.channel_bars, params.percentile_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.streak_percentile < 1.0:
            raise ValueError("streak_percentile must be between zero and one")
        if not np.isfinite(params.elastic_atr) or params.elastic_atr <= 0:
            raise ValueError("elastic_atr must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")

        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["channel_low"] = data["low"].rolling(params.channel_bars).min().shift(1)
        out["channel_high"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["position"] = (
            (data["close"] - out["channel_low"])
            / out["channel_width"].where(out["channel_width"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["position_change"] = out["position"].diff()
        out["declining"] = out["position_change"].lt(0).fillna(False)
        out["episode"] = (~out["declining"]).cumsum()
        out["streak"] = out["declining"].astype(int).groupby(out["episode"]).cumsum()
        out["close_change"] = data["close"].diff()
        out["deformation"] = (
            out["close_change"].where(out["declining"], 0.0)
            .groupby(out["episode"]).cumsum()
        )
        out["streak_threshold"] = (
            out["streak"].where(out["position_change"].notna())
            .rolling(params.percentile_bars)
            .quantile(params.streak_percentile).shift(1)
        )
        out["entry"] = (
            out["position_change"].gt(0)
            & out["close_change"].gt(0)
            & out["streak"].shift(1).ge(2)
            & out["streak"].shift(1).ge(out["streak_threshold"])
            & out["deformation"].shift(1).lt(0)
            & out["deformation"].shift(1).ge(-params.elastic_atr * out["atr"].shift(1))
            & out["atr"].gt(0)
            & out["atr"].notna()
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticStreakParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        active = False
        held = 0
        stop_price = np.nan

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Freeze the stop at the next-open entry using the decision-bar ATR.
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
                    active = True
                    held = 0
            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < stop_price or held >= 2:
                    active = False
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
            decisions[i] = int(active or pending)

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 97.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False],
        "atr": [1.0, 10.0, 10.0, 10.0, 10.0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = ElasticStreakParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 98.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
