from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSnrBandParams:
    vol_window: int = 20
    atr_window: int = 14
    gap_lower: float = 0.5
    gap_upper: float = 2.5
    atr_mult: float = 2.0
    target_vol: float = 0.008
    max_hold: int = 2
    min_size: float = 0.25
    max_size: float = 2.0
    scale_floor: float = 0.6
    scale_span: float = 0.8


class GeneratedStrategy(BaseStrategy[GapSnrBandParams]):
    strategy_id = "gen_a1_1778886363"

    @classmethod
    def params_type(cls):
        return GapSnrBandParams

    @staticmethod
    def warmup_bars(params: GapSnrBandParams) -> int:
        return int(max(params.vol_window, params.atr_window)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapSnrBandParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        prev_close = close.shift(1)
        log_ret = np.log(close / prev_close)
        gap_ret = np.log(open_ / prev_close)

        realized_vol = log_ret.rolling(
            params.vol_window, min_periods=params.vol_window
        ).std()
        safe_vol = realized_vol.replace(0.0, np.nan)

        snr = gap_ret / safe_vol

        tr_components = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()

        base_size = params.target_vol / safe_vol

        span = params.gap_upper - params.gap_lower
        if span <= 0.0:
            span = 1e-9
        snr_norm = ((snr - params.gap_lower) / span).clip(lower=0.0, upper=1.0)
        signal_scale = params.scale_floor + params.scale_span * snr_norm

        out = pd.DataFrame(index=data.index)
        out["gap_ret"] = gap_ret
        out["realized_vol"] = realized_vol
        out["snr"] = snr
        out["atr"] = atr
        out["base_size"] = base_size
        out["signal_scale"] = signal_scale
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSnrBandParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        snr = indicators["snr"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        base_size = indicators["base_size"].to_numpy(dtype=float)
        signal_scale = indicators["signal_scale"].to_numpy(dtype=float)

        entry = (
            np.isfinite(snr)
            & np.isfinite(atr)
            & np.isfinite(base_size)
            & (atr > 0.0)
            & (snr >= params.gap_lower)
            & (snr <= params.gap_upper)
        )

        raw_signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        in_pos = False
        stop_level = 0.0
        bars_in = 0
        trade_size = 1.0
        max_hold = int(params.max_hold) if params.max_hold > 0 else 1

        for i in range(n):
            if not in_pos:
                if entry[i]:
                    in_pos = True
                    bars_in = 0
                    stop_level = close[i] - params.atr_mult * atr[i]
                    raw = base_size[i] * signal_scale[i]
                    if not np.isfinite(raw):
                        raw = params.min_size
                    trade_size = float(
                        min(max(raw, params.min_size), params.max_size)
                    )
                    raw_signal[i] = 1
                    size[i] = trade_size
                else:
                    raw_signal[i] = 0
            else:
                bars_in += 1
                if close[i] <= stop_level or bars_in >= max_hold:
                    in_pos = False
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
                    size[i] = trade_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=params.min_size, upper=params.max_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
