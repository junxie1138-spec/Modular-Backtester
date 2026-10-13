from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CongestionParams:
    baseline_window: int = 20
    rank_window: int = 60
    congestion_cutoff: float = 0.7
    return_tail: float = 0.2
    pressure_floor: float = 0.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CongestionParams]):
    strategy_id = "gen_local_1790964434"

    @classmethod
    def params_type(cls) -> type[CongestionParams]:
        return CongestionParams

    def warmup_bars(self, params: CongestionParams) -> int:
        return int(params.baseline_window + params.rank_window)

    def indicators(self, data: pd.DataFrame, params: CongestionParams) -> pd.DataFrame:
        if params.baseline_window < 2 or params.rank_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["relative_range"] = span / close
        out["range_baseline"] = out["relative_range"].rolling(params.baseline_window).mean()
        out["volume_baseline"] = volume.rolling(params.baseline_window).mean()
        out["congestion"] = (volume / out["volume_baseline"]) / (out["relative_range"] / out["range_baseline"])
        out["intraday_return"] = close / opening - 1.0
        out["congestion_rank"] = out["congestion"].rolling(params.rank_window).rank(pct=True)
        out["return_rank"] = out["intraday_return"].rolling(params.rank_window).rank(pct=True)
        out["closing_pressure"] = (2.0 * close - data["high"] - data["low"]) / span
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CongestionParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        if not 0.5 < params.congestion_cutoff < 1.0 or not 0.0 < params.return_tail < 0.5:
            raise ValueError("Invalid percentile thresholds")
        if not -1.0 <= params.pressure_floor <= 1.0:
            raise ValueError("Invalid closing-pressure threshold")
        congested = indicators["congestion_rank"] >= params.congestion_cutoff
        fluid = indicators["congestion_rank"] <= 1.0 - params.congestion_cutoff
        reversal = congested & (indicators["return_rank"] <= params.return_tail) & (indicators["intraday_return"] < 0.0)
        continuation = fluid & (indicators["return_rank"] >= 1.0 - params.return_tail) & (indicators["intraday_return"] > 0.0)
        pressure_agrees = indicators["closing_pressure"] >= params.pressure_floor
        ready = indicators[["congestion_rank", "return_rank", "intraday_return", "closing_pressure"]].notna().all(axis=1)
        entries = ((reversal | continuation) & pressure_agrees & ready).to_numpy(dtype=bool)
        positions = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for bar in range(len(data)):
            if entry_bar >= 0:
                if bar - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    positions[bar] = 1
            elif entries[bar]:
                entry_bar = bar
                positions[bar] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 99.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"congestion_rank": 0.9, "return_rank": 0.1, "intraday_return": -0.01, "closing_pressure": 0.8}, index=index)
    strategy = GeneratedStrategy()
    for holding, expected in ((1, [0, 1, 0, 1, 0, 1, 0]), (2, [0, 1, 1, 0, 1, 1, 0])):
        result = strategy.generate_signals(data, derived, None, CongestionParams(hold_bars=holding))
        assert result.data["signal"].tolist() == expected
    derived["closing_pressure"] = np.nan
    assert strategy.generate_signals(data, derived, None, CongestionParams()).data["signal"].eq(0).all()
