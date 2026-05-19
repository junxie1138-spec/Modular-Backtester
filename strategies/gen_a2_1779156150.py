from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 30
    atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """Volatility-clustering regime onset (autocorrelation family).

    The lag-1 autocorrelation of absolute returns is a direct measure of
    volatility clustering. It oscillates between an 'elastic' regime (value
    <= 0: magnitude shocks do not persist, noise) and a 'plastic' regime
    (value > 0: shocks deform the process persistently). Entry fires on the
    yield point - the fresh crossing from elastic into plastic - provided
    price is above its own rolling mean. Exit is an ATR rolling-high trailing
    stop that only ratchets up.
    """

    strategy_id = "gen_a2_1779156150"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.lookback) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        n = max(int(params.lookback), 2)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(n, min_periods=n).mean()

        ret = close.pct_change()
        absret = ret.abs()
        # Volatility autocorrelation: rolling lag-1 autocorr of |returns|.
        vac = absret.rolling(n, min_periods=n).corr(absret.shift(1))

        sma = close.rolling(n, min_periods=n).mean()
        trend_ok = (close > sma).astype(float)

        vac_prev = vac.shift(1)
        # Fresh crossing from the elastic regime (<=0) into the plastic
        # regime (>0). NaN comparisons evaluate False, so this is NaN-safe.
        cross_up = ((vac > 0.0) & (vac_prev <= 0.0)).astype(float)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["vac"] = vac
        out["trend_ok"] = trend_ok
        out["cross_up"] = cross_up
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        k = float(params.atr_mult)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        trend_ok = indicators["trend_ok"].to_numpy(dtype=float)
        cross_up = indicators["cross_up"].to_numpy(dtype=float)

        n = len(close)
        pos = np.zeros(n, dtype=np.int64)

        in_pos = False
        hwm = 0.0
        for i in range(n):
            a = atr[i]
            if not in_pos:
                if (
                    cross_up[i] == 1.0
                    and trend_ok[i] == 1.0
                    and np.isfinite(a)
                    and a > 0.0
                    and np.isfinite(close[i])
                ):
                    in_pos = True
                    hwm = close[i]
                    pos[i] = 1
            else:
                c = close[i]
                if np.isfinite(c) and c > hwm:
                    hwm = c
                if not np.isfinite(a):
                    in_pos = False
                    pos[i] = 0
                    continue
                stop = hwm - k * a
                if np.isfinite(c) and c <= stop:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
