from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    mom_window: int = 5
    vol_window: int = 20
    rank_window: int = 100
    enter_pct: float = 0.85
    ma_window: int = 200
    regime_buffer: float = 0.0
    target_vol: float = 0.15
    vol_lookback: int = 20
    profit_target: float = 0.03
    time_stop: int = 2
    max_size: float = 1.5
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779151993"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        rank_need = params.mom_window + params.vol_window + params.rank_window
        return int(max(params.ma_window, rank_need, params.vol_lookback)) + 5

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        p = params
        close = data["close"].astype(float)
        ret = close.pct_change()

        mom = close.pct_change(max(1, p.mom_window))
        rvol = ret.rolling(max(2, p.vol_window)).std()
        denom = (rvol * np.sqrt(max(1, p.mom_window))).replace(0.0, np.nan)
        risk_adj = mom / denom
        risk_adj = risk_adj.replace([np.inf, -np.inf], np.nan)
        rank = risk_adj.rolling(max(2, p.rank_window)).rank(pct=True)

        ma = close.rolling(max(2, p.ma_window)).mean()
        regime = pd.Series(0.0, index=close.index)
        upper = ma * (1.0 + p.regime_buffer)
        lower = ma * (1.0 - p.regime_buffer)
        regime[close > upper] = 1.0
        regime[close < lower] = -1.0
        regime[ma.isna()] = np.nan

        ann_vol = ret.rolling(max(2, p.vol_lookback)).std() * np.sqrt(252.0)
        vol_size = p.target_vol / ann_vol.replace(0.0, np.nan)
        vol_size = vol_size.replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out["risk_adj_rank"] = rank
        out["regime"] = regime
        out["vol_size"] = vol_size
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        p = params
        close = data["close"].to_numpy(dtype=float)
        rank = indicators["risk_adj_rank"].to_numpy(dtype=float)
        regime = indicators["regime"].to_numpy(dtype=float)
        n = len(close)
        sig = np.zeros(n, dtype=int)

        enter_hi = float(p.enter_pct)
        enter_lo = 1.0 - float(p.enter_pct)
        time_stop = max(1, int(p.time_stop))
        profit_target = float(p.profit_target)

        pos = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            r = rank[i]
            rp = rank[i - 1] if i > 0 else np.nan
            reg = regime[i]

            if pos == 0:
                if np.isnan(r) or np.isnan(rp) or np.isnan(reg):
                    sig[i] = 0
                    continue
                long_trigger = (r >= enter_hi) and (rp < enter_hi) and (reg > 0.0)
                short_trigger = (r <= enter_lo) and (rp > enter_lo) and (reg < 0.0)
                if long_trigger:
                    pos = 1
                    entry_price = close[i]
                    bars_held = 0
                    sig[i] = 1
                elif short_trigger:
                    pos = -1
                    entry_price = close[i]
                    bars_held = 0
                    sig[i] = -1
                else:
                    sig[i] = 0
            else:
                bars_held += 1
                if entry_price > 0.0:
                    pnl = (close[i] / entry_price - 1.0) * pos
                else:
                    pnl = 0.0
                if pnl >= profit_target or bars_held >= time_stop:
                    pos = 0
                    entry_price = 0.0
                    bars_held = 0
                    sig[i] = 0
                else:
                    sig[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        )

        size = indicators["vol_size"].copy()
        size = size.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
        size = size.clip(lower=float(p.min_size), upper=float(p.max_size))
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
