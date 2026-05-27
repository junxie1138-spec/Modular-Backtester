from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringTensionRegimeParams:
    short_vol_window: int = 5
    long_vol_window: int = 60
    short_ma_window: int = 5
    atr_window: int = 14
    regime_ma_window: int = 200
    vol_ratio_percentile_window: int = 252
    vol_ratio_percentile_threshold: float = 0.25
    tension_atr_threshold: float = 0.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SpringTensionRegimeParams]):
    strategy_id = "gen_a1_1779883149"

    @classmethod
    def params_type(cls):
        return SpringTensionRegimeParams

    def warmup_bars(self, params: SpringTensionRegimeParams) -> int:
        return int(max(
            params.regime_ma_window,
            params.vol_ratio_percentile_window + params.long_vol_window + 2,
            params.atr_window + 2,
            params.short_ma_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: SpringTensionRegimeParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        prev_close = close.shift(1)

        ret = close.pct_change()

        short_vol = ret.rolling(params.short_vol_window, min_periods=params.short_vol_window).std()
        long_vol = ret.rolling(params.long_vol_window, min_periods=params.long_vol_window).std()
        long_vol_safe = long_vol.where(long_vol > 0, np.nan)
        vol_ratio = short_vol / long_vol_safe

        vol_ratio_pct = vol_ratio.rolling(
            params.vol_ratio_percentile_window,
            min_periods=params.vol_ratio_percentile_window,
        ).rank(pct=True)

        tr_components = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr_safe = atr.where(atr > 0, np.nan)

        short_ma = close.rolling(params.short_ma_window, min_periods=params.short_ma_window).mean()
        regime_ma = close.rolling(params.regime_ma_window, min_periods=params.regime_ma_window).mean()

        tension = (short_ma - close) / atr_safe

        bull_regime = (close > regime_ma).astype(float)

        out = pd.DataFrame(
            {
                "short_vol": short_vol,
                "long_vol": long_vol,
                "vol_ratio": vol_ratio,
                "vol_ratio_pct": vol_ratio_pct,
                "atr": atr,
                "short_ma": short_ma,
                "regime_ma": regime_ma,
                "tension": tension,
                "bull_regime": bull_regime,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringTensionRegimeParams,
    ) -> SignalFrame:
        vol_compressed = indicators["vol_ratio_pct"] <= params.vol_ratio_percentile_threshold
        spring_loaded = indicators["tension"] >= params.tension_atr_threshold
        bull = indicators["bull_regime"] > 0.5

        entry_mask = (vol_compressed & spring_loaded & bull).fillna(False).to_numpy()

        n = len(entry_mask)
        hold_bars = max(1, int(params.hold_bars))
        raw_signal = np.zeros(n, dtype=np.int64)
        remaining = 0
        for i in range(n):
            if remaining > 0:
                raw_signal[i] = 1
                remaining -= 1
                continue
            if entry_mask[i]:
                raw_signal[i] = 1
                remaining = hold_bars - 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index, dtype="int64")
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
