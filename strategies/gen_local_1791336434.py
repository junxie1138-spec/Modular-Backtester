from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SessionPressureParams:
    rank_window: int = 20
    tail: float = 0.25


class GeneratedStrategy(BaseStrategy[SessionPressureParams]):
    strategy_id = "gen_local_1791336434"

    @classmethod
    def params_type(cls) -> type[SessionPressureParams]:
        return SessionPressureParams

    def warmup_bars(self, params: SessionPressureParams) -> int:
        if params.rank_window < 4:
            raise ValueError("rank_window must be at least 4")
        if not 0.0 < params.tail < 0.5:
            raise ValueError("tail must be between zero and one half")
        return int(params.rank_window + 4)

    def indicators(self, data: pd.DataFrame, params: SessionPressureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["overnight"] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        out["session_difference"] = out["intraday"] - out["overnight"]
        out["gross_motion"] = out["intraday"].abs() + out["overnight"].abs()
        out["daily_motion"] = out["intraday"] + out["overnight"]
        out["difference_sum"] = out["session_difference"].rolling(4, min_periods=4).sum()
        out["gross_sum"] = out["gross_motion"].rolling(4, min_periods=4).sum()
        out["net_motion"] = out["daily_motion"].rolling(4, min_periods=4).sum()
        out["pressure"] = out["difference_sum"] / out["gross_sum"].where(out["gross_sum"] > 0)
        out["percentile"] = out["pressure"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SessionPressureParams) -> SignalFrame:
        self.warmup_bars(params)
        valid = np.isfinite(indicators[["pressure", "percentile", "net_motion"]]).all(axis=1)
        long_entry = valid & (indicators["percentile"] <= params.tail) & (indicators["pressure"] < 0) & (indicators["net_motion"] < 0)
        short_entry = valid & (indicators["percentile"] >= 1.0 - params.tail) & (indicators["pressure"] > 0) & (indicators["net_motion"] > 0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, (buy, sell) in enumerate(zip(long_entry.to_numpy(), short_entry.to_numpy())):
            if position == 1:
                if sell:
                    position = 0
            elif position == -1:
                if buy:
                    position = 0
            elif buy:
                position = 1
            elif sell:
                position = -1
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "pressure": [np.nan, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0],
        "percentile": [np.nan, 0.1, 0.1, 0.5, 0.9, 0.9, 0.5, 0.1],
        "net_motion": [np.nan, -0.01, -0.01, 0.0, 0.01, 0.01, 0.0, -0.01]
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, SessionPressureParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1]
    assert (result["size"] > 0).all()
