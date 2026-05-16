from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VarianceRatioParams:
    vr_q: int = 5
    vr_window: int = 63
    vr_threshold: float = 1.05
    ma_period: int = 200


class GeneratedStrategy(BaseStrategy[VarianceRatioParams]):
    strategy_id = "gen_a1_1778913544"

    @classmethod
    def params_type(cls) -> type[VarianceRatioParams]:
        return VarianceRatioParams

    @staticmethod
    def warmup_bars(params: VarianceRatioParams) -> int:
        # 200MA is the longest lookback; VR needs window + q + 1 bars.
        return int(params.ma_period + params.vr_window + params.vr_q + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VarianceRatioParams) -> pd.DataFrame:
        close = data["close"]

        # Close-to-close returns at 1-period and q-period horizons.
        r1 = close / close.shift(1) - 1.0
        rq = close / close.shift(params.vr_q) - 1.0

        var1 = r1.rolling(params.vr_window).var()
        varq = rq.rolling(params.vr_window).var()

        # Variance ratio: > 1 implies positive autocorrelation / persistence.
        denom = params.vr_q * var1
        denom = denom.where(denom > 0.0, np.nan)
        vr = varq / denom
        vr = vr.replace([np.inf, -np.inf], np.nan)

        ma = close.rolling(params.ma_period).mean()

        out = pd.DataFrame(index=data.index)
        out["vr"] = vr
        out["ma200"] = ma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VarianceRatioParams,
    ) -> SignalFrame:
        close = data["close"]
        vr = indicators["vr"]
        ma = indicators["ma200"]

        # Entry condition: persistence regime AND bullish 200MA regime filter.
        persistence = vr > params.vr_threshold
        regime_ok = close > ma
        entry = (persistence & regime_ok).fillna(False)

        # Long-only. Signal-reversal exit: signal returns to 0 the moment the
        # entry condition flips false (VR falls back through threshold or
        # price loses the 200MA). No separate stop is applied.
        raw_signal = entry.astype(int)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
