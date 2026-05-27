from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 20
    si_window: int = 20
    regime_ma: int = 200
    susceptible_z: float = 0.5
    infection_z: float = 1.5
    growth_threshold: float = 0.04
    fresh_lookback: int = 3
    hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779880726"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return int(max(params.regime_ma, params.ma_window + params.si_window) + 5)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        std = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        std_safe = std.where(std > 0.0, np.nan)
        z = (close - ma) / std_safe

        regime_ma = close.rolling(params.regime_ma, min_periods=params.regime_ma).mean()

        abs_z = z.abs()
        susceptible = (abs_z < params.susceptible_z).astype(float)
        infected = (abs_z > params.infection_z).astype(float)

        s_count = susceptible.rolling(params.si_window, min_periods=params.si_window).sum()
        i_count = infected.rolling(params.si_window, min_periods=params.si_window).sum()

        denom = float(params.si_window * params.si_window)
        growth = (s_count * i_count) / denom

        up_extreme = (z > params.infection_z).astype(float)
        dn_extreme = (z < -params.infection_z).astype(float)
        fresh_lk = max(1, int(params.fresh_lookback))
        fresh_up = up_extreme.rolling(fresh_lk, min_periods=1).max()
        fresh_dn = dn_extreme.rolling(fresh_lk, min_periods=1).max()

        return pd.DataFrame({
            "z": z,
            "regime_ma": regime_ma,
            "growth": growth,
            "s_count": s_count,
            "i_count": i_count,
            "fresh_up": fresh_up,
            "fresh_dn": fresh_dn,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        close = data["close"].astype(float)
        growth = indicators["growth"]
        regime_ma = indicators["regime_ma"]
        fresh_up = indicators["fresh_up"]
        fresh_dn = indicators["fresh_dn"]

        above_regime = (close > regime_ma).fillna(False)
        below_regime = (close < regime_ma).fillna(False)
        growth_ok = (growth > params.growth_threshold).fillna(False)
        up_now = (fresh_up > 0).fillna(False)
        dn_now = (fresh_dn > 0).fillna(False)

        long_trigger = (growth_ok & up_now & above_regime).to_numpy(dtype=bool)
        short_trigger = (growth_ok & dn_now & below_regime).to_numpy(dtype=bool)

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)
        position = 0
        bars_left = 0
        hold = max(1, int(params.hold_bars))

        for i in range(n):
            if position != 0:
                raw[i] = position
                bars_left -= 1
                if bars_left <= 0:
                    position = 0
                    bars_left = 0
            else:
                if long_trigger[i]:
                    position = 1
                    bars_left = hold - 1
                    raw[i] = 1
                elif short_trigger[i]:
                    position = -1
                    bars_left = hold - 1
                    raw[i] = -1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
