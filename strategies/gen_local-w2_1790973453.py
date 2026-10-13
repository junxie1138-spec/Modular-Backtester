from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    ma_bars: int = 20
    compression_bars: int = 5
    percentile_bars: int = 126
    compression_quantile: float = 0.30
    tail_quantile: float = 0.20
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local-w2_1790973453"

    @classmethod
    def params_type(cls):
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(max(
            params.ma_bars + params.percentile_bars + 1,
            params.compression_bars + params.percentile_bars,
            params.ma_bars + params.hold_bars,
        ))

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if not (
            params.ma_bars >= 2 and params.compression_bars >= 1
            and params.percentile_bars >= 2 and 3 <= params.hold_bars <= 5
            and 0 < params.compression_quantile < 1
            and 0 < params.tail_quantile < 0.5 and params.profit_target > 0
        ):
            raise ValueError("Invalid lookbacks, percentiles, target, or holding period")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma"] = close.rolling(params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars).std(ddof=0).replace(0, np.nan)
        out["z"] = (close - out["ma"]) / out["std"]
        out["compression"] = (
            (data["high"] - data["low"]) / close
        ).rolling(params.compression_bars).mean()
        out["compression_cutoff"] = out["compression"].shift(1).rolling(
            params.percentile_bars
        ).quantile(params.compression_quantile)
        out["tail_cutoff"] = out["z"].shift(1).rolling(
            params.percentile_bars
        ).quantile(params.tail_quantile)
        out["anchor_drift"] = out["ma"].diff(params.hold_bars).abs()
        out["displacement"] = (close - out["ma"]).abs()
        out["entry"] = (
            (out["compression"] <= out["compression_cutoff"])
            & (out["z"].shift(1) < out["tail_cutoff"].shift(1))
            & (out["z"] >= out["tail_cutoff"])
            & (out["z"] > out["z"].shift(1))
            & (out["z"] < 0)
            & (out["anchor_drift"] < out["displacement"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ElasticParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
            if active:
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if target_hit or expired:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": [100.0, 100.0, 102.0, 100.0, 100.0,
                  100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, True,
                  False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = ElasticParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    assert not strategy.indicators(data, params)["entry"].any()
