from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778896573"

    # Hardcoded epidemic / climatology constants (intentionally NOT tunable).
    _CLIM = 252        # climatological window for the volatility z-score
    _CLIM_MIN = 60     # min periods before the climatology is usable
    _INFLECT = 0.5     # logistic inflection point = SI epidemic peak
    _SIZE_LO = 0.5     # vol-target size floor
    _SIZE_HI = 1.5     # vol-target size cap

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        w = max(int(params.vol_window), 2)
        return int(GeneratedStrategy._CLIM + w + 5)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        w = max(int(params.vol_window), 2)
        clim = self._CLIM
        cmin = self._CLIM_MIN

        close = data["close"].astype(float)
        ret = close.pct_change()

        # Realized volatility = the infection load.
        vol = ret.rolling(w).std()

        # Climatological mean/std of vol -> logistic-squashed infected fraction.
        mu = vol.rolling(clim, min_periods=cmin).mean()
        sd = vol.rolling(clim, min_periods=cmin).std()
        sd = sd.where(sd > 0.0)
        z = (vol - mu) / sd
        z = z.clip(-10.0, 10.0)
        infected = 1.0 / (1.0 + np.exp(-z))          # I in (0, 1)

        # New infections per bar = velocity of the SI epidemic curve.
        new_cases = infected.diff()

        # Volatility-target size: inverse vol vs its own climatological median.
        vol_ref = vol.rolling(clim, min_periods=cmin).median()
        size = vol_ref / vol.where(vol > 0.0)
        size = size.clip(self._SIZE_LO, self._SIZE_HI).fillna(1.0)

        out = pd.DataFrame(index=data.index)
        out["vol"] = vol
        out["infected"] = infected
        out["new_cases"] = new_cases
        out["size"] = size
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        infected = indicators["infected"]
        new_cases = indicators["new_cases"]
        prev_cases = new_cases.shift(1)

        # SI epidemic PEAK: new cases roll over (+ -> -) while infection is high.
        peak = (new_cases < 0.0) & (prev_cases >= 0.0) & (infected > self._INFLECT)
        # SI epidemic TROUGH: new cases turn up (- -> +) while infection is low.
        trough = (new_cases > 0.0) & (prev_cases <= 0.0) & (infected < self._INFLECT)

        raw = np.where(peak.to_numpy(), 1,
                       np.where(trough.to_numpy(), -1, 0)).astype(int)

        # Fixed-bar exit: hold exactly N bars after entry, no signal-based exit.
        hold_n = max(int(params.hold_bars), 1)
        pos = np.zeros(n, dtype=int)
        hold = 0
        direction = 0
        for t in range(n):
            if hold > 0:
                pos[t] = direction
                hold -= 1
                continue
            if raw[t] != 0:
                direction = int(raw[t])
                pos[t] = direction
                hold = hold_n - 1

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)

        size = indicators["size"].astype(float)
        size = size.where(size > 0.0, 1.0).fillna(1.0)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
