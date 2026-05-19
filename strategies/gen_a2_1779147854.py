from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779147854"

    # Hardcoded constants - kept off the tunable surface to honor the
    # <=2 tunable-param twist.
    _ATR_WINDOW = 14
    _Z_ENTRY = -1.0
    _MAX_HOLD = 5
    _SUSCEPT_FRAC = 0.5
    _TARGET_VOL = 0.15
    _SIZE_MIN = 0.10
    _SIZE_MAX = 1.50

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(int(params.lookback), GeneratedStrategy._ATR_WINDOW)) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        lb = max(int(params.lookback), 2)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(lb).mean()
        sd = close.rolling(lb).std()
        z = (close - ma) / sd.replace(0.0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self._ATR_WINDOW).mean()

        ret = close.pct_change()
        rv = ret.rolling(lb).std() * np.sqrt(252.0)

        # Susceptible pool: fraction of the recent window spent below the MA.
        susceptible = (z < 0.0).astype(float).rolling(lb).mean()

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        out["atr"] = atr
        out["rv"] = rv
        out["susceptible"] = susceptible
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        rv = indicators["rv"].to_numpy(dtype=float)
        susc = indicators["susceptible"].to_numpy(dtype=float)

        z_prev = np.empty(n, dtype=float)
        z_prev[0] = np.nan
        if n > 1:
            z_prev[1:] = z[:-1]

        atr_mult = float(params.atr_mult)

        signal = np.zeros(n, dtype=int)
        position = 0
        hwm = np.nan
        bars_held = 0

        for i in range(n):
            if position == 0:
                cross_up = (
                    np.isfinite(z_prev[i])
                    and np.isfinite(z[i])
                    and z_prev[i] < self._Z_ENTRY
                    and z[i] >= self._Z_ENTRY
                )
                susceptible_enough = (
                    np.isfinite(susc[i]) and susc[i] >= self._SUSCEPT_FRAC
                )
                tradable = np.isfinite(atr[i]) and atr[i] > 0.0
                if cross_up and susceptible_enough and tradable:
                    position = 1
                    hwm = close[i]
                    bars_held = 0
                    signal[i] = 1
            else:
                bars_held += 1
                if np.isfinite(close[i]) and close[i] > hwm:
                    hwm = close[i]
                stop = -np.inf
                if np.isfinite(atr[i]) and atr[i] > 0.0:
                    stop = hwm - atr_mult * atr[i]
                if (close[i] < stop) or (bars_held >= self._MAX_HOLD):
                    position = 0
                    hwm = np.nan
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        # Volatility-targeted sizing: hold risk roughly constant.
        with np.errstate(divide="ignore", invalid="ignore"):
            size = self._TARGET_VOL / rv
        size = np.where(np.isfinite(size) & (rv > 0.0), size, 1.0)
        size = np.clip(size, self._SIZE_MIN, self._SIZE_MAX)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size.astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
