from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TransferParams:
    scale_window: int = 10
    threshold: float = 0.4


class GeneratedStrategy(BaseStrategy[TransferParams]):
    strategy_id = "gen_local-w2_1791279301"

    @classmethod
    def params_type(cls) -> type[TransferParams]:
        return TransferParams

    @staticmethod
    def warmup_bars(params: TransferParams) -> int:
        if params.scale_window < 2:
            raise ValueError("scale_window must be at least 2")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        return int(params.scale_window + 4)

    @staticmethod
    def indicators(data: pd.DataFrame, params: TransferParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        result["previous_close"] = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(data["open"] > 0)
        result["gap_return"] = opening / result["previous_close"] - 1.0
        result["intraday_return"] = data["close"].where(data["close"] > 0) / opening - 1.0
        result["gap_acceleration"] = result["gap_return"].diff()
        result["intraday_acceleration"] = result["intraday_return"].diff()
        result["transfer"] = result["intraday_acceleration"] - result["gap_acceleration"]
        result["scale"] = result["transfer"].abs().rolling(
            params.scale_window, min_periods=params.scale_window
        ).mean().shift(1)
        result["score"] = result["transfer"] / result["scale"].where(result["scale"] > 0)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TransferParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        score = indicators["score"].reindex(data.index)
        valid = np.isfinite(score)
        direction = pd.Series(
            np.select(
                [valid & score.gt(params.threshold), valid & score.lt(-params.threshold)],
                [1, -1],
                default=0,
            ),
            index=data.index,
            dtype=int,
        )
        confirmed = direction.eq(direction.shift(1)) & direction.ne(0)
        positions = np.zeros(len(data), dtype=int)
        state = 0
        for i, (current, entry) in enumerate(zip(direction.to_numpy(), confirmed.to_numpy())):
            if state != 0 and current == -state:
                state = 0
            elif state == 0 and entry:
                state = int(current)
            positions[i] = state
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    sample = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    derived = pd.DataFrame(
        {"score": [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0]},
        index=index,
    )
    signals = GeneratedStrategy.generate_signals(sample, derived, None, TransferParams()).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, -1, 0]
    assert signals["size"].gt(0).all()
    flat = GeneratedStrategy.indicators(sample, TransferParams(scale_window=2))
    assert GeneratedStrategy.generate_signals(sample, flat, None, TransferParams(scale_window=2)).data["signal"].eq(0).all()
