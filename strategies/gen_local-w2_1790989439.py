from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityRecoveryParams:
    graph_bars: int = 16
    peak_bars: int = 40
    min_drawdown: float = 0.02
    min_connectivity: float = 0.40
    min_bullish_share: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityRecoveryParams]):
    strategy_id = "gen_local-w2_1790989439"

    @classmethod
    def params_type(cls):
        return VisibilityRecoveryParams

    def warmup_bars(self, params: VisibilityRecoveryParams) -> int:
        return int(max(params.graph_bars, params.peak_bars, 2) + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityRecoveryParams) -> pd.DataFrame:
        graph_bars = max(3, int(params.graph_bars))
        peak_bars = max(2, int(params.peak_bars))
        close = data["close"].astype(float).where(data["close"] > 0)
        log_close = np.log(close)
        count = len(data)
        degree = np.zeros(count, dtype=float)
        bullish_degree = np.zeros(count, dtype=float)
        minimum_slope = np.full(count, np.inf)

        # An edge to lag d exists iff its slope is below every nearer slope.
        # This is the strict natural-visibility condition for intermediate prices.
        for lag in range(1, graph_bars):
            slope = ((log_close - log_close.shift(lag)) / lag).to_numpy()
            visible = np.isfinite(slope) & (slope < minimum_slope)
            degree += visible
            bullish_degree += visible & (slope > 0.0)
            minimum_slope = np.minimum(minimum_slope, np.where(np.isfinite(slope), slope, np.inf))

        graph_ready = log_close.rolling(graph_bars, min_periods=graph_bars).count().eq(graph_bars).to_numpy()
        degree = np.where(graph_ready, degree, np.nan)
        bullish_degree = np.where(graph_ready, bullish_degree, np.nan)
        peak = close.rolling(peak_bars, min_periods=peak_bars).max()

        out = pd.DataFrame(index=data.index)
        out["log_close"] = log_close
        out["peak"] = peak
        out["drawdown"] = 1.0 - close / peak
        out["endpoint_degree"] = degree
        out["bullish_degree"] = bullish_degree
        out["connectivity"] = degree / float(graph_bars - 1)
        out["bullish_share"] = bullish_degree / np.maximum(degree, 1.0)
        out["log_return"] = log_close.diff()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityRecoveryParams,
    ) -> SignalFrame:
        setup = (
            indicators["drawdown"].ge(params.min_drawdown)
            & indicators["connectivity"].ge(params.min_connectivity)
            & indicators["bullish_share"].ge(params.min_bullish_share)
            & indicators["log_return"].gt(0.0)
        ).fillna(False)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entered_at = -1
        hold_bars = max(1, int(params.hold_bars))
        first_decision = self.warmup_bars(params) - 1

        for i in range(len(data)):
            if entered_at >= 0:
                if i - entered_at < hold_bars:
                    signal[i] = 1
                else:
                    # Force a flat target at expiry, even if entry remains confirmed.
                    entered_at = -1
            elif i >= first_decision and confirmed[i]:
                entered_at = i
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
