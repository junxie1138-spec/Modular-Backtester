from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicRecoveryParams:
    ma_len: int = 200
    dd_lookback: int = 40
    dd_threshold: float = 0.03
    roc_len: int = 5
    accel_len: int = 3
    atr_len: int = 14
    atr_k: float = 2.5
    max_hold: int = 5
    size_base: float = 1.0
    dd_size_scale: float = 3.0
    size_cap: float = 2.0


class GeneratedStrategy(BaseStrategy[EpidemicRecoveryParams]):
    """Drawdown-recovery long-only strategy.

    Treats a pullback as an epidemic curve. Entry fires when the rate-of-change
    acceleration (second difference of ROC) crosses from non-positive to
    positive -- the inflection point of the recovery curve -- while price sits
    in a deep enough drawdown (large 'susceptible pool') and the long-term
    200-day regime is bullish. Exit is a FIXED ATR volatility stop measured
    from the entry price, with a hard time cap to keep the 3-5 day horizon.
    """

    strategy_id = "gen_a1_1778907510"

    @classmethod
    def params_type(cls) -> type[EpidemicRecoveryParams]:
        return EpidemicRecoveryParams

    @staticmethod
    def warmup_bars(params: EpidemicRecoveryParams) -> int:
        return int(max(
            params.ma_len,
            params.dd_lookback,
            params.atr_len,
            params.roc_len + params.accel_len,
        )) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: EpidemicRecoveryParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
        roll_max = close.rolling(params.dd_lookback, min_periods=params.dd_lookback).max()
        dd = close / roll_max - 1.0

        roc = close.pct_change(params.roc_len)
        accel = roc.diff(params.accel_len)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        out = pd.DataFrame(index=data.index)
        out["ma"] = ma
        out["dd"] = dd
        out["accel"] = accel
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EpidemicRecoveryParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        dd = indicators["dd"].to_numpy(dtype=float)
        accel = indicators["accel"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        entry_atr = 0.0
        pos_size = 1.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                stop = entry_price - params.atr_k * entry_atr
                exit_now = (close[i] < stop) or (bars_held >= params.max_hold)
                if exit_now:
                    in_pos = False
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size[i] = pos_size
                continue

            if i == 0:
                continue

            valid = (
                not np.isnan(ma[i])
                and not np.isnan(dd[i])
                and not np.isnan(accel[i])
                and not np.isnan(accel[i - 1])
                and not np.isnan(atr[i])
            )
            if not valid or atr[i] <= 0.0:
                continue

            regime_ok = close[i] > ma[i]
            deep_enough = dd[i] <= -params.dd_threshold
            inflection = (accel[i] > 0.0) and (accel[i - 1] <= 0.0)

            if regime_ok and deep_enough and inflection:
                in_pos = True
                entry_price = close[i]
                entry_atr = atr[i]
                bars_held = 0
                depth = abs(dd[i])
                raw = params.size_base + params.dd_size_scale * depth
                pos_size = float(min(max(raw, 0.1), params.size_cap))
                signal[i] = 1
                size[i] = pos_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
