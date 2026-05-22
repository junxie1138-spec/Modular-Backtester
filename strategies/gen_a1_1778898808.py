from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InterferenceParams:
    atr_period: int = 14
    gap_atr_mult: float = 0.25
    body_atr_mult: float = 0.25
    stop_atr_mult: float = 2.5
    max_hold_bars: int = 18
    vol_period: int = 20
    target_vol: float = 0.15
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[InterferenceParams]):
    strategy_id = "gen_a1_1778898808"

    @classmethod
    def params_type(cls) -> type[InterferenceParams]:
        return InterferenceParams

    @staticmethod
    def warmup_bars(params: InterferenceParams) -> int:
        return int(max(params.atr_period, params.vol_period)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: InterferenceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        prev_close = close.shift(1)

        # ATR (simple mean of true range) - drives both thresholds and the stop
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["atr"] = atr

        # two return half-cycles
        gap = open_ - prev_close          # overnight wave
        body = close - open_              # intraday wave
        out["gap"] = gap
        out["body"] = body

        # two-primitive AND: both half-cycles up and each beyond an ATR-scaled
        # magnitude threshold -> constructive interference (in-phase antinode)
        gap_thr = params.gap_atr_mult * atr
        body_thr = params.body_atr_mult * atr
        gap_up = gap > gap_thr
        body_up = body > body_thr
        entry_raw = (gap_up & body_up).fillna(False)
        out["entry_raw"] = entry_raw.astype(bool)

        # realized volatility for volatility-targeted sizing
        log_ret = np.log(close / prev_close)
        realized_vol = log_ret.rolling(
            params.vol_period, min_periods=params.vol_period
        ).std() * np.sqrt(252.0)
        out["realized_vol"] = realized_vol

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InterferenceParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_raw = indicators["entry_raw"].to_numpy()
        realized_vol = indicators["realized_vol"].to_numpy(dtype=float)

        position = np.zeros(n, dtype=int)
        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                exit_now = False
                # fixed volatility-stop: stop_level fixed at entry, not trailed
                if not np.isnan(close[i]) and close[i] < stop_level:
                    exit_now = True
                if bars_held >= int(params.max_hold_bars):
                    exit_now = True
                if exit_now:
                    in_pos = False
                    bars_held = 0
                    position[i] = 0
                else:
                    position[i] = 1
            else:
                a = atr[i]
                if bool(entry_raw[i]) and not np.isnan(a) and a > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    stop_level = entry_price - params.stop_atr_mult * a
                    bars_held = 0
                    position[i] = 1
                else:
                    position[i] = 0

        # volatility-targeting: scale exposure toward a constant target vol
        rv = np.where(
            np.isnan(realized_vol) | (realized_vol <= 0.0),
            params.target_vol,
            realized_vol,
        )
        size = params.target_vol / rv
        size = np.clip(size, params.min_size, params.max_size)

        df = pd.DataFrame(index=idx)
        df["signal"] = position
        df["size"] = size.astype(float)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
