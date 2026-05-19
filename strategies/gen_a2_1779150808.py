from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_period: int = 200
    atr_period: int = 14
    tr_avg_period: int = 20
    tr_mult: float = 1.2
    bias_threshold: float = 0.05
    clv_entry: float = 0.5
    stop_k: float = 2.5
    max_hold: int = 10
    size_value: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779150808"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.ma_period,
                       params.atr_period + 1,
                       params.tr_avg_period + 1)) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        ma = close.rolling(params.ma_period).mean()

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()
        avg_tr = tr.rolling(params.tr_avg_period).mean()

        hl = (high - low).replace(0.0, np.nan)
        clv = ((2.0 * close - high - low) / hl).fillna(0.0)

        month = pd.Series(data.index.month, index=data.index)
        csum = clv.groupby(month).cumsum()
        ccnt = clv.groupby(month).cumcount() + 1
        seasonal_bias = (csum / ccnt).fillna(0.0)

        ind = pd.DataFrame(index=data.index)
        ind["ma"] = ma
        ind["tr"] = tr
        ind["atr"] = atr
        ind["avg_tr"] = avg_tr
        ind["clv"] = clv
        ind["seasonal_bias"] = seasonal_bias
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        warmup = self.warmup_bars(params)

        close = data["close"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        tr = indicators["tr"].to_numpy(dtype=float)
        avg_tr = indicators["avg_tr"].to_numpy(dtype=float)
        clv = indicators["clv"].to_numpy(dtype=float)
        bias = indicators["seasonal_bias"].to_numpy(dtype=float)

        allow_short = bool(ctx.allow_short) if hasattr(ctx, "allow_short") else True

        raw = np.zeros(n, dtype=int)
        pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if (i < warmup or np.isnan(ma[i]) or np.isnan(atr[i])
                    or np.isnan(avg_tr[i]) or np.isnan(tr[i]) or atr[i] <= 0.0):
                raw[i] = pos if pos != 0 else 0
                if pos != 0:
                    bars_held += 1
                continue

            if pos == 0:
                range_expansion = tr[i] > params.tr_mult * avg_tr[i]
                long_ok = (bias[i] >= params.bias_threshold
                           and close[i] > ma[i]
                           and range_expansion
                           and clv[i] >= params.clv_entry)
                short_ok = (allow_short
                            and bias[i] <= -params.bias_threshold
                            and close[i] < ma[i]
                            and range_expansion
                            and clv[i] <= -params.clv_entry)
                if long_ok:
                    pos = 1
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
                    raw[i] = 1
                elif short_ok:
                    pos = -1
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
                    raw[i] = -1
                else:
                    raw[i] = 0
            else:
                bars_held += 1
                exit_now = False
                if pos == 1:
                    if close[i] <= entry_price - params.stop_k * entry_atr:
                        exit_now = True
                else:
                    if close[i] >= entry_price + params.stop_k * entry_atr:
                        exit_now = True
                if bars_held >= params.max_hold:
                    exit_now = True
                if exit_now:
                    pos = 0
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        size = float(params.size_value)
        if size <= 0.0:
            size = 1.0
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
