from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnVoteParams:
    vote_window: int = 4
    vol_window: int = 20
    target_vol: float = 0.12
    vol_floor: float = 0.01
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ReturnVoteParams]):
    strategy_id = "gen_local-w2_1791302532"

    @classmethod
    def params_type(cls) -> type[ReturnVoteParams]:
        return ReturnVoteParams

    @staticmethod
    def warmup_bars(params: ReturnVoteParams) -> int:
        return int(max(params.vote_window, params.vol_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ReturnVoteParams) -> pd.DataFrame:
        if params.vote_window < 1 or params.vol_window < 2:
            raise ValueError("Invalid rolling window")
        if not all(np.isfinite(x) and x > 0 for x in (params.target_vol, params.vol_floor, params.max_size)) or params.max_size > 1:
            raise ValueError("Invalid volatility target or exposure limit")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: x > 0)
        out["returns"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_vote"] = np.sign(out["returns"]) * np.sqrt(out["returns"].abs())
        out["vote_balance"] = out["return_vote"].rolling(params.vote_window, min_periods=params.vote_window).sum()
        out["annual_vol"] = out["returns"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=1) * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["annual_vol"].clip(lower=params.vol_floor)).clip(lower=1e-8, upper=params.max_size)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnVoteParams) -> SignalFrame:
        score = indicators["vote_balance"]
        ready = score.notna() & indicators["annual_vol"].notna()
        # Equal and opposite zero crossings govern entry and exit.
        # Exact ties and unavailable observations preserve the existing state.
        state = pd.Series(np.nan, index=data.index, dtype=float)
        state.loc[ready & score.gt(0)] = 1.0
        state.loc[ready & score.lt(0)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = state.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["target_size"].shift(1).ffill().fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = ReturnVoteParams()
    idx = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": np.full(8, 100.0)}, index=idx)
    ind = pd.DataFrame({"vote_balance": [np.nan, 1.0, 2.0, 0.0, -1.0, -2.0, 1.0, 0.0], "annual_vol": 0.2, "target_size": 0.6}, index=idx)
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1]
    assert np.allclose(result["size"], [0.95] + [0.6] * 7)
    assert GeneratedStrategy.warmup_bars(p) == 21
