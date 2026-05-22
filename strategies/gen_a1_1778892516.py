from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_regime: int = 200
    dd_lookback: int = 60
    min_drawdown: float = 0.03
    atr_period: int = 14
    vol_baseline: int = 100
    recovery_frac: float = 0.5
    trail_atr_mult: float = 3.0
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778892516"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.ma_regime,
            params.atr_period + params.vol_baseline,
            params.dd_lookback,
            params.atr_period,
        )) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        ma = close.rolling(params.ma_regime, min_periods=params.ma_regime).mean()

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        atr_pct = atr / close.replace(0.0, np.nan)

        # Volatility 'infection' = ATR% in excess of its rolling-median baseline.
        vol_base = atr_pct.rolling(
            params.vol_baseline, min_periods=params.vol_baseline
        ).median()
        excess = (atr_pct - vol_base).clip(lower=0.0)
        peak_excess = excess.rolling(
            params.dd_lookback, min_periods=params.dd_lookback
        ).max()

        # Infected fraction: 1.0 at peak infection, decays toward 0 as it burns out.
        infected = excess / peak_excess.replace(0.0, np.nan)

        peak = close.rolling(
            params.dd_lookback, min_periods=params.dd_lookback
        ).max()
        drawdown = close / peak.replace(0.0, np.nan) - 1.0

        ind = pd.DataFrame(index=data.index)
        ind["ma"] = ma
        ind["atr"] = atr
        ind["infected"] = infected
        ind["drawdown"] = drawdown
        return ind

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        infected = indicators["infected"].to_numpy(dtype=float)
        drawdown = indicators["drawdown"].to_numpy(dtype=float)

        n = len(close)

        prev_inf = np.concatenate(([np.nan], infected[:-1]))
        # Fresh downward cross of the infected fraction = volatility burnout.
        cross = (infected < params.recovery_frac) & (prev_inf >= params.recovery_frac)
        in_dd = drawdown <= -params.min_drawdown
        regime = close > ma  # 200-day MA bull-regime gate; NaN -> False
        atr_ok = np.isfinite(atr) & (atr > 0.0)
        entry_raw = cross & in_dd & regime & atr_ok

        raw = np.zeros(n, dtype=int)
        in_pos = False
        hwm = 0.0
        for i in range(n):
            if not in_pos:
                if entry_raw[i]:
                    in_pos = True
                    hwm = close[i]
                    raw[i] = 1
            else:
                if close[i] > hwm:
                    hwm = close[i]
                if not np.isfinite(atr[i]):
                    in_pos = False
                    raw[i] = 0
                    continue
                stop = hwm - params.trail_atr_mult * atr[i]
                if close[i] <= stop:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = float(max(params.base_size, 1e-6))
        return SignalFrame(data=df, signal_column="signal", size_column="size")
