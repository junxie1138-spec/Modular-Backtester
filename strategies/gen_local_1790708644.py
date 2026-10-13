from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    vol_window: int = 20
    shock_z: float = 1.3
    energy_share: float = 0.72
    profit_target: float = 0.009
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790708644"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return max(params.vol_window + 1, 9)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        r = np.log(data["close"].div(data["close"].shift(1))).replace([np.inf, -np.inf], np.nan)
        vol = r.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        e2 = (r - r.shift(1)).pow(2) / 2
        e4 = (r.rolling(2).sum() - r.shift(2).rolling(2).sum()).pow(2) / 4
        e8 = (r.rolling(4).sum() - r.shift(4).rolling(4).sum()).pow(2) / 8
        share = (e2 + e4).div((e2 + e4 + e8).replace(0, np.nan))
        shock = (-r).div(vol.replace(0, np.nan))
        out = pd.DataFrame({"return": r, "vol": vol, "energy_2": e2, "energy_4": e4, "energy_8": e8, "fine_share": share, "shock": shock}, index=data.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["stake"] = (0.15 + 0.3 * out["shock"] * out["fine_share"]).clip(lower=0.15, upper=0.95)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        candidate = ((indicators["return"] < 0) & (indicators["shock"] >= params.shock_z) & (indicators["fine_share"] >= params.energy_share)).to_numpy()
        stakes = indicators["stake"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.15, dtype=float)
        active = False
        fill_index = -1
        entry_price = 0.0
        stake = 0.15

        for i in range(len(data)):
            if active:
                if i == fill_index:
                    entry_price = opens[i]
                if closes[i] >= entry_price * (1 + params.profit_target) or i - fill_index + 1 >= params.max_hold_bars:
                    active = False
                else:
                    signal[i] = 1
                    size[i] = stake
            elif candidate[i]:
                active = True
                fill_index = i + 1
                stake = stakes[i]
                signal[i] = 1
                size[i] = stake

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
