from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoverySlopeParams:
    peak_window: int = 20
    slope_span: int = 4
    depth_pct: float = 0.015
    rebound_pct: float = 0.004


class GeneratedStrategy(BaseStrategy[RecoverySlopeParams]):
    strategy_id = "gen_local-w2_1791301466"

    @classmethod
    def params_type(cls) -> type[RecoverySlopeParams]:
        return RecoverySlopeParams

    def warmup_bars(self, params: RecoverySlopeParams) -> int:
        return max(params.peak_window + 5, params.slope_span + 3)

    def indicators(self, data: pd.DataFrame, params: RecoverySlopeParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        log_close = np.log(close)
        change = log_close.diff()
        slope = change.ewm(span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
        drawdown = close.div(close.rolling(params.peak_window, min_periods=params.peak_window).max()).sub(1)
        out = pd.DataFrame(index=data.index)
        out["innovation"] = change.sub(slope.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["drawdown"] = drawdown.fillna(0.0)
        out["rebound"] = drawdown.sub(drawdown.rolling(5, min_periods=5).min()).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoverySlopeParams,
    ) -> SignalFrame:
        innovation = indicators["innovation"]
        up = innovation.gt(0)
        down = innovation.lt(0)
        up_two = up & up.shift(1, fill_value=False)
        down_two = down & down.shift(1, fill_value=False)
        recovery = indicators["drawdown"].le(-params.depth_pct) & indicators["rebound"].ge(params.rebound_pct)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(self.warmup_bars(params), len(data)):
            if position == 1 and down_two.iat[i]:
                position = 0
            elif position == -1 and up_two.iat[i]:
                position = 0
            elif position == 0 and recovery.iat[i]:
                if up_two.iat[i]:
                    position = 1
                elif down_two.iat[i]:
                    position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
