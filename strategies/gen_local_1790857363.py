from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAreaParams:
    history: int = 126
    percentile: float = 0.85
    profit_target: float = 0.025
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GapAreaParams]):
    strategy_id = "gen_local_1790857363"

    @classmethod
    def params_type(cls):
        return GapAreaParams

    def warmup_bars(self, params: GapAreaParams) -> int:
        return int(params.history + 4)

    def indicators(self, data: pd.DataFrame, params: GapAreaParams) -> pd.DataFrame:
        if params.history < 3 or not 0 < params.percentile < 1:
            raise ValueError("history must be >= 3 and percentile between 0 and 1")
        if params.profit_target <= 0 or params.max_hold < 1:
            raise ValueError("profit_target and max_hold must be positive")
        gap_ratio = data["open"] / data["close"].shift(1)
        body_ratio = data["close"] / data["open"]
        gap = np.log(gap_ratio.where(gap_ratio > 0)).replace([np.inf, -np.inf], np.nan)
        body = np.log(body_ratio.where(body_ratio > 0)).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame({"gap": gap, "body": body}, index=data.index)
        out["gap_old"] = gap.shift(2)
        out["gap_middle"] = gap.shift(1)
        out["body_old"] = body.shift(2)
        out["body_middle"] = body.shift(1)
        # Level-two antisymmetric signature of three (gap, body) increments.
        out["area"] = 0.5 * (
            out["gap_old"] * out["body_middle"]
            - out["body_old"] * out["gap_middle"]
            + out["gap_old"] * body
            - out["body_old"] * gap
            + out["gap_middle"] * body
            - out["body_middle"] * gap
        )
        out["prior_gap_sum"] = out["gap_old"] + out["gap_middle"]
        out["abs_area"] = out["area"].abs()
        out["area_threshold"] = out["abs_area"].rolling(
            params.history, min_periods=params.history
        ).quantile(params.percentile).shift(1)
        extreme = out["abs_area"] > out["area_threshold"]
        long_setup = (
            extreme & (out["area"] < 0) & (out["prior_gap_sum"] < 0)
            & (gap > 0) & (body > 0)
        )
        short_setup = (
            extreme & (out["area"] > 0) & (out["prior_gap_sum"] > 0)
            & (gap < 0) & (body < 0)
        )
        out["setup"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAreaParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            if side != 0:
                if entry_bar == -1:
                    # An entry decided at the preceding close fills at this open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = side * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_up = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_up:
                    side = 0
                    entry_bar = -1
                    continue
            elif setups[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(setups[i])
                entry_bar = -1
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=18, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[[1, 7]], "close"] = [103.0, 97.0]
    indicators = pd.DataFrame({"setup": np.zeros(18, dtype=int)}, index=index)
    indicators.loc[index[[0, 6, 12]], "setup"] = [1, -1, 1]
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, GapAreaParams(max_hold=2)).data
    expected = np.zeros(18, dtype=int)
    expected[[1, 7, 13, 14]] = [1, -1, 1, 1]
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result.index.equals(index) and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
