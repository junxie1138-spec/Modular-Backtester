from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolTermRegimeParams:
    short_vol_window: int = 10
    long_vol_window: int = 60
    ma_window: int = 50
    atr_window: int = 14
    compression_thresh: float = 0.85
    expansion_thresh: float = 1.25
    spike_atr_mult: float = 2.5
    refractory_bars: int = 5
    confirm_bars: int = 2
    profit_target: float = 0.06
    time_stop_bars: int = 18


class GeneratedStrategy(BaseStrategy[VolTermRegimeParams]):
    strategy_id = "gen_a1_1778883358"

    @classmethod
    def params_type(cls) -> type[VolTermRegimeParams]:
        return VolTermRegimeParams

    @staticmethod
    def warmup_bars(params: VolTermRegimeParams) -> int:
        base = max(
            params.long_vol_window + 1,
            params.ma_window,
            params.atr_window + 1,
        )
        return int(base + params.confirm_bars + params.refractory_bars + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolTermRegimeParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        returns = close.pct_change()
        short_vol = returns.rolling(params.short_vol_window).std()
        long_vol = returns.rolling(params.long_vol_window).std()
        # Volatility term-structure slope: short realized vol vs long realized vol.
        vol_ratio = short_vol / long_vol.replace(0.0, np.nan)
        ma = close.rolling(params.ma_window).mean()

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()

        bar_move = (close - prev_close).abs()
        spike = bar_move > (params.spike_atr_mult * atr)

        ind = pd.DataFrame(index=data.index)
        ind["vol_ratio"] = vol_ratio
        ind["ma"] = ma
        ind["atr"] = atr
        ind["spike"] = spike.fillna(False).astype(float)
        ind["close"] = close
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolTermRegimeParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)
        close = data["close"].to_numpy(dtype=float)
        vol_ratio = indicators["vol_ratio"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        spike = indicators["spike"].to_numpy(dtype=float)

        # Raw regime-conditioned directional signal.
        # Compressed vol regime -> momentum continuation.
        # Expanded vol regime  -> mean-reversion fade.
        raw = np.zeros(n, dtype=int)
        for i in range(n):
            vr = vol_ratio[i]
            m = ma[i]
            c = close[i]
            if not np.isfinite(vr) or not np.isfinite(m):
                continue
            if vr < params.compression_thresh:
                raw[i] = 1 if c > m else -1
            elif vr > params.expansion_thresh:
                raw[i] = -1 if c > m else 1
            else:
                raw[i] = 0

        # Two-bar confirmation: identical non-zero raw signal must persist
        # for confirm_bars consecutive bars before it counts as an entry.
        cb = max(1, int(params.confirm_bars))
        confirmed = np.zeros(n, dtype=int)
        for i in range(cb - 1, n):
            v = raw[i]
            if v == 0:
                continue
            ok = True
            for k in range(1, cb):
                if raw[i - k] != v:
                    ok = False
                    break
            if ok:
                confirmed[i] = v

        # Position walk with profit-target + time-stop exit and a
        # post-spike refractory window that suppresses new entries.
        signal = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        bars_held = 0
        last_spike = -(10 ** 9)
        for i in range(n):
            if spike[i] == 1.0:
                last_spike = i

            exited = False
            if position != 0:
                bars_held += 1
                pnl = (close[i] - entry_price) / entry_price * position
                if pnl >= params.profit_target or bars_held >= params.time_stop_bars:
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    exited = True

            if position == 0 and not exited and confirmed[i] != 0:
                in_refractory = (i - last_spike) <= params.refractory_bars
                if not in_refractory:
                    position = confirmed[i]
                    entry_price = close[i]
                    bars_held = 0

            signal[i] = position

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
