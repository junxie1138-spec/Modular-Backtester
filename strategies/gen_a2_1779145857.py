from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Fixed (non-tunable) structural constants - the twist caps tunable params at 2.
_DISP_WINDOW = 10   # bars summed into the spring displacement
_VOL_WINDOW = 60    # bars used to normalize displacement by return volatility
_TIME_STOP = 5      # max holding horizon in bars (3-5 day target)


@dataclass(slots=True)
class GeneratedParams:
    entry_z: float = 1.0      # normalized spring-stretch threshold to cross
    target_pct: float = 0.04  # profit target as a fraction of entry price


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779145857"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # pct_change consumes 1 bar; vol and displacement windows stack on top.
        return _VOL_WINDOW + _DISP_WINDOW + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()
        # Spring displacement: net stretch of close-to-close returns over the window.
        displacement = ret.rolling(_DISP_WINDOW).sum()
        # Volatility scale: expected magnitude of a random-walk stretch of this length.
        vol = ret.rolling(_VOL_WINDOW).std()
        scale = (vol * np.sqrt(_DISP_WINDOW)).replace(0.0, np.nan)
        z = displacement / scale
        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["z"] = z
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        z = np.nan_to_num(indicators["z"].to_numpy(dtype=float), nan=0.0)
        ret = np.nan_to_num(indicators["ret"].to_numpy(dtype=float), nan=0.0)

        thr = float(params.entry_z)
        target = float(params.target_pct)

        raw = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        bars_held = 0

        # Path-dependent profit-target + time-stop exit; bar loop is the clean form.
        for i in range(1, n):
            if position == 0:
                # Spring stretches up through the threshold and is still loading
                # (latest close-to-close return pushes in the same direction).
                if z[i] > thr and z[i - 1] <= thr and ret[i] > 0.0:
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
            else:
                bars_held += 1
                gain = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if gain >= target or bars_held >= _TIME_STOP:
                    position = 0
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
