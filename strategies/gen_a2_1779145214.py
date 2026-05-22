from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    phase_cap: int = 21
    season_lookback: int = 12
    rank_window: int = 60
    upper_pct: float = 0.80
    lower_pct: float = 0.20
    min_snr: float = 0.05
    profit_target: float = 0.03
    time_stop: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779145214"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.phase_cap * params.season_lookback
                   + params.rank_window + params.time_stop + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        idx = data.index
        close = data["close"].astype(float)

        # trading-day-of-month phase (1-indexed, capped)
        period = pd.Series(pd.PeriodIndex(idx, freq="M"), index=idx)
        phase = period.groupby(period).cumcount() + 1
        cap = max(int(params.phase_cap), 1)
        phase = phase.clip(upper=cap).astype(int)

        ret1 = close.pct_change()

        lb = max(int(params.season_lookback), 2)
        tmp = pd.DataFrame({"ret1": ret1, "phase": phase.to_numpy()}, index=idx)
        # per-phase trailing mean / std over recent occurrences of that phase
        season_mean = tmp.groupby("phase")["ret1"].transform(
            lambda s: s.rolling(lb, min_periods=3).mean())
        season_std = tmp.groupby("phase")["ret1"].transform(
            lambda s: s.rolling(lb, min_periods=3).std())

        eps = 1e-6
        snr = season_mean / (season_std + eps)
        snr = snr.replace([np.inf, -np.inf], np.nan)

        rw = max(int(params.rank_window), 5)
        rank_pct = snr.rolling(rw, min_periods=max(rw // 2, 3)).rank(pct=True)

        out = pd.DataFrame(index=idx)
        out["phase"] = phase.astype(float)
        out["snr"] = snr.fillna(0.0)
        out["rank_pct"] = rank_pct.fillna(0.5)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        idx = data.index
        close = data["close"].astype(float).to_numpy()
        snr = indicators["snr"].to_numpy()
        rank_pct = indicators["rank_pct"].to_numpy()

        msnr = float(params.min_snr)
        # strong phase: high cross-phase rank AND positive seasonal SNR
        bull = (rank_pct > float(params.upper_pct)) & (snr > msnr)
        # weak phase: low cross-phase rank AND negative seasonal SNR
        bear = (rank_pct < float(params.lower_pct)) & (snr < -msnr)

        bull_prev = np.concatenate(([False], bull[:-1]))
        bear_prev = np.concatenate(([False], bear[:-1]))
        # two-bar confirmation before entry
        long_entry = bull & bull_prev
        short_entry = bear & bear_prev

        n = len(close)
        pos = np.zeros(n, dtype=np.int64)
        pt = float(params.profit_target)
        tstop = max(int(params.time_stop), 1)

        state = 0
        entry_price = 0.0
        held = 0
        for i in range(n):
            if state == 0:
                if long_entry[i]:
                    state = 1
                    entry_price = close[i]
                    held = 0
                elif short_entry[i]:
                    state = -1
                    entry_price = close[i]
                    held = 0
                pos[i] = state
            else:
                held += 1
                if entry_price <= 0.0 or close[i] <= 0.0:
                    gain = 0.0
                elif state == 1:
                    gain = close[i] / entry_price - 1.0
                else:
                    gain = entry_price / close[i] - 1.0
                # profit-target OR time-stop, whichever fires first
                if gain >= pt or held >= tstop:
                    state = 0
                    pos[i] = 0
                else:
                    pos[i] = state

        df = pd.DataFrame(index=idx)
        df["signal"] = pos
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
