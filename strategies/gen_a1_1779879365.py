from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    autocorr_window: int = 5
    rank_window: int = 4
    rank_threshold: float = 0.75
    min_positive_sum: float = 1.0
    hold_bars: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779879365"

    @classmethod
    def params_type(cls) -> type[GenA1Params]:
        return GenA1Params

    @classmethod
    def warmup_bars(cls, params: GenA1Params) -> int:
        aw = max(1, int(params.autocorr_window))
        rw = max(1, int(params.rank_window))
        # +1 for the lag-1 shift inside the sign product
        return aw + rw + 1

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        aw = max(1, int(params.autocorr_window))
        rw = max(1, int(params.rank_window))

        close = data["close"].astype(float)
        returns = close.pct_change()

        # Sign of each return, NaN-safe (first bar has no prior return).
        sign = np.sign(returns.fillna(0.0))

        # Lag-1 sign product: positive when consecutive returns agree in direction,
        # negative when they disagree. Short-horizon serial-correlation proxy.
        lag1_product = sign * sign.shift(1)

        # "Queue fill" / capacity gauge: sum of lag-1 agreements over a tiny window.
        autocorr_score = lag1_product.rolling(aw, min_periods=aw).sum()

        # Self-referential rank: where does today's capacity gauge sit inside its
        # own recent rw-bar distribution? pct=True gives a value in (0, 1].
        rank_pct = autocorr_score.rolling(rw, min_periods=rw).rank(pct=True)

        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["autocorr_score"] = autocorr_score
        out["rank_pct"] = rank_pct
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=np.int64)

        rank_pct = indicators["rank_pct"].to_numpy()
        score = indicators["autocorr_score"].to_numpy()

        threshold = float(params.rank_threshold)
        min_sum = float(params.min_positive_sum)
        hold = max(1, int(params.hold_bars))

        in_position_until = -1
        for i in range(n):
            if i <= in_position_until:
                raw[i] = 1
                continue

            r = rank_pct[i]
            s = score[i]
            if (
                not np.isnan(r)
                and not np.isnan(s)
                and r >= threshold
                and s >= min_sum
            ):
                raw[i] = 1
                in_position_until = i + hold - 1
            else:
                raw[i] = 0

        size_val = float(params.size)
        if not np.isfinite(size_val) or size_val <= 0.0:
            size_val = 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = size_val
        return SignalFrame(data=df, signal_column="signal", size_column="size")
