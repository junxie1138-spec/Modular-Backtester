from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    lookback: int = 20
    gap_coherence: float = 0.55


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = "gen_local_1791215557"

    @classmethod
    def params_type(cls) -> type[GapRecoveryParams]:
        return GapRecoveryParams

    @staticmethod
    def warmup_bars(params: GapRecoveryParams) -> int:
        if params.lookback < 2 or not 0.0 < params.gap_coherence < 1.0:
            raise ValueError("lookback must be >= 2 and gap_coherence between 0 and 1")
        return int(max(params.lookback + 1, 5))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = data["open"] / out["prior_close"] - 1.0
        out["net_gap"] = out["gap"].rolling(4, min_periods=4).sum()
        out["gross_gap"] = out["gap"].abs().rolling(4, min_periods=4).sum()
        out["coherence"] = out["net_gap"] / out["gross_gap"].replace(0.0, np.nan)
        out["intraday"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        out["peak"] = data["close"].rolling(params.lookback, min_periods=params.lookback).max()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["prior_close"]).abs(),
             (data["low"] - out["prior_close"]).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["mean_range"] = out["true_range"].rolling(
            params.lookback, min_periods=params.lookback
        ).mean()
        out["drawdown"] = out["peak"] - data["close"]
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRecoveryParams,
    ) -> SignalFrame:
        ready = indicators[["coherence", "intraday", "drawdown", "mean_range"]].notna().all(axis=1)
        stressed = ready & (indicators["drawdown"] > indicators["mean_range"])
        long_entry = stressed & (indicators["coherence"] >= params.gap_coherence) & (indicators["intraday"] > 0)
        short_entry = stressed & (indicators["coherence"] <= -params.gap_coherence) & (indicators["intraday"] < 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Opposite entry exits to flat; a later bar may enter again.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "coherence": [0.8, np.nan, 0.1, -0.8, -0.8, 0.1, 0.8, 0.8],
        "intraday": [0.01, 0.0, 0.0, -0.01, -0.01, 0.0, 0.01, 0.01],
        "drawdown": 3.0,
        "mean_range": 1.0,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapRecoveryParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert result.index.equals(index) and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
