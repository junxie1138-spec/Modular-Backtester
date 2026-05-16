from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class LoadedSpringParams:
    rank_window: int = 50
    ma_window: int = 200
    low_pct: float = 0.25
    high_pct: float = 0.80
    tension_decay: float = 0.85
    tension_min: float = 2.5
    profit_target: float = 0.03
    max_hold: int = 2
    size_base: float = 1.0
    size_tension_scale: float = 0.15


class GeneratedStrategy(BaseStrategy[LoadedSpringParams]):
    strategy_id = "gen_a1_1778895325"

    @classmethod
    def params_type(cls):
        return LoadedSpringParams

    @staticmethod
    def warmup_bars(params: LoadedSpringParams) -> int:
        return int(max(params.ma_window, params.rank_window + 1)) + 5

    def indicators(self, data: pd.DataFrame, params: LoadedSpringParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        n = len(close)
        rw = max(int(params.rank_window), 2)
        mw = max(int(params.ma_window), 2)

        # Position of close within its own trailing window, as a percentile in (0, 1].
        pct_rank = close.rolling(rw, min_periods=rw).rank(pct=True)
        sma = close.rolling(mw, min_periods=mw).mean()

        # Compression flag: close pinned in the lower band of its range-percentile.
        comp = (pct_rank < float(params.low_pct)).fillna(False).to_numpy()

        # Leaky-integrator 'spring tension': accumulates while compressed, bleeds off otherwise.
        decay = float(params.tension_decay)
        if not np.isfinite(decay) or decay < 0.0:
            decay = 0.0
        if decay > 0.999:
            decay = 0.999
        tension = np.zeros(n, dtype=float)
        acc = 0.0
        for i in range(n):
            acc = acc * decay + (1.0 if comp[i] else 0.0)
            tension[i] = acc
        tension_s = pd.Series(tension, index=close.index)
        # Tension as of the prior bar = energy stored by the compression that precedes the breakout.
        tension_lag = tension_s.shift(1).fillna(0.0)

        # Release trigger: percentile rank crosses up into its upper band.
        prev_rank = pct_rank.shift(1)
        breakout = (pct_rank >= float(params.high_pct)) & (prev_rank < float(params.high_pct))
        breakout = breakout.fillna(False)

        # Hard twist: only act in a bull regime above the 200-day MA.
        regime_ok = (close > sma).fillna(False)

        entry_cond = breakout & regime_ok & (tension_lag >= float(params.tension_min))

        entry_size = (float(params.size_base) +
                      float(params.size_tension_scale) * tension_lag).clip(0.5, 2.5)

        out = pd.DataFrame(index=close.index)
        out["pct_rank"] = pct_rank
        out["sma"] = sma
        out["tension"] = tension_s
        out["tension_lag"] = tension_lag
        out["breakout"] = breakout.astype(float)
        out["regime_ok"] = regime_ok.astype(float)
        out["entry_cond"] = entry_cond.astype(bool)
        out["entry_size"] = entry_size.astype(float)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: LoadedSpringParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)
        entry = indicators["entry_cond"].fillna(False).to_numpy()
        size_in = indicators["entry_size"].fillna(1.0).to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        pt = float(params.profit_target)
        mh = max(int(params.max_hold), 1)

        position = 0
        entry_price = 0.0
        bars_held = 0
        cur_size = 1.0

        # Path-dependent exit: profit-target OR time-stop, whichever fires first.
        for i in range(n):
            if position == 0:
                if bool(entry[i]):
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                    cur_size = float(size_in[i])
                    if not np.isfinite(cur_size) or cur_size <= 0.0:
                        cur_size = 1.0
                    signal[i] = 1
                    size[i] = cur_size
            else:
                bars_held += 1
                if entry_price > 0.0:
                    gain = close[i] / entry_price - 1.0
                else:
                    gain = 0.0
                if gain >= pt or bars_held >= mh:
                    position = 0
                    signal[i] = 0
                    size[i] = 1.0
                else:
                    signal[i] = 1
                    size[i] = cur_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
