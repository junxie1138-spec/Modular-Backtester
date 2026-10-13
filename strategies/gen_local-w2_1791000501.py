from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureBreadthParams:
    path_window: int = 20
    breadth_window: int = 20
    area_threshold: float = 0.08
    breadth_threshold: float = 0.20


class GeneratedStrategy(BaseStrategy[SignatureBreadthParams]):
    strategy_id = "gen_local-w2_1791000501"

    @classmethod
    def params_type(cls) -> type[SignatureBreadthParams]:
        return SignatureBreadthParams

    def warmup_bars(self, params: SignatureBreadthParams) -> int:
        return int(max(2, params.path_window, params.breadth_window) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureBreadthParams) -> pd.DataFrame:
        n = max(2, int(params.path_window))
        b = max(2, int(params.breadth_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        out["log_return"] = out["log_close"].diff()
        out["signed_return"] = np.sign(out["log_return"])
        out["path_variation"] = out["log_return"].abs().rolling(n, min_periods=n).sum()
        origin = out["log_close"].shift(n)
        past_sum = out["log_close"].shift(1).rolling(n, min_periods=n).sum()
        # Twice the antisymmetric level-two signature of (session time, log price).
        out["signature_area"] = (
            (n - 1) * (out["log_close"] - origin)
            - 2.0 * (past_sum - n * origin)
        )
        denominator = (n - 1) * out["path_variation"]
        out["normalized_area"] = out["signature_area"].div(denominator.where(denominator > 0))
        out["return_breadth"] = out["signed_return"].rolling(b, min_periods=b).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureBreadthParams,
    ) -> SignalFrame:
        area = indicators["normalized_area"]
        breadth = indicators["return_breadth"]
        valid = area.notna() & breadth.notna()
        area_threshold = max(0.0, float(params.area_threshold))
        breadth_threshold = max(0.0, float(params.breadth_threshold))
        long_entry = valid & (area > area_threshold) & (breadth > breadth_threshold)
        short_entry = valid & (area < -area_threshold) & (breadth < -breadth_threshold)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        state = 0
        for i, candidate in enumerate(candidates):
            # Only an opposite two-primitive entry closes an existing position.
            # The reversal bar is flat; a subsequent qualifying bar can re-enter.
            if state != 0 and candidate == -state:
                state = 0
            elif state == 0 and candidate != 0:
                state = int(candidate)
            positions[i] = state
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
