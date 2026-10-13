from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CancellationSqueezeParams:
    squeeze_bars: int = 6
    baseline_bars: int = 30
    compression_ratio: float = 0.55
    cancellation_max: float = 0.25
    release_multiple: float = 1.8
    profit_target: float = 0.012
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[CancellationSqueezeParams]):
    strategy_id = "gen_local-w2_1790986043"

    @classmethod
    def params_type(cls) -> type[CancellationSqueezeParams]:
        return CancellationSqueezeParams

    def warmup_bars(self, params: CancellationSqueezeParams) -> int:
        return int(max(200, params.baseline_bars + 2, params.squeeze_bars + 2))

    def indicators(self, data: pd.DataFrame, params: CancellationSqueezeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        squared = out["return"].pow(2)
        out["squeeze_rms"] = np.sqrt(squared.rolling(params.squeeze_bars, min_periods=params.squeeze_bars).mean())
        out["baseline_rms"] = np.sqrt(squared.rolling(params.baseline_bars, min_periods=params.baseline_bars).mean())
        out["return_sum"] = out["return"].rolling(params.squeeze_bars, min_periods=params.squeeze_bars).sum()
        out["absolute_return_sum"] = out["return"].abs().rolling(params.squeeze_bars, min_periods=params.squeeze_bars).sum()
        out["cancellation"] = out["return_sum"].abs().div(out["absolute_return_sum"].replace(0.0, np.nan))
        out["compression"] = out["squeeze_rms"].div(out["baseline_rms"].replace(0.0, np.nan))
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["compressed"] = (
            out["compression"].le(params.compression_ratio)
            & out["cancellation"].le(params.cancellation_max)
            & out["squeeze_rms"].gt(0.0)
        ).fillna(False)
        out["prior_compressed"] = out["compressed"].shift(1, fill_value=False)
        out["prior_squeeze_rms"] = out["squeeze_rms"].shift(1)
        out["release"] = (
            out["prior_compressed"]
            & out["return"].abs().ge(params.release_multiple * out["prior_squeeze_rms"])
        ).fillna(False)
        out["long_entry"] = (out["release"] & out["return"].gt(0.0) & close.gt(out["ma200"])).fillna(False)
        out["short_entry"] = (out["release"] & out["return"].lt(0.0) & close.lt(out["ma200"])).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CancellationSqueezeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        position = 0
        decision_bar = -1
        entry_price = np.nan
        first_bar = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if position != 0:
                # The preceding close's decision fills at this bar's open.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        continue
                age = i - decision_bar
                gain = position * (closes[i] / entry_price - 1.0)
                if age >= params.max_hold_bars or (np.isfinite(gain) and gain >= params.profit_target):
                    position = 0
                    continue
                raw[i] = position
            elif i >= first_bar and np.isfinite(closes[i]) and closes[i] > 0.0:
                direction = 1 if longs[i] else (-1 if shorts[i] else 0)
                if direction != 0:
                    position = direction
                    decision_bar = i
                    entry_price = np.nan
                    raw[i] = position
        df["signal"] = raw
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
