from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    inf_window: int = 10
    infection_q: float = 0.25
    entry_pct: float = 0.30
    atr_period: int = 14
    atr_stop_mult: float = 2.5
    trail_mult: float = 1.5
    breakeven_pct: float = 1.0
    target_vol: float = 0.15


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778914092"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # chain: pct_change(5) + rolling(rank_window) x2 + rolling(inf_window)
        return 2 * params.rank_window + params.inf_window + 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret1 = close.pct_change(1)
        ret3 = close.pct_change(3)
        ret5 = close.pct_change(5)

        W = params.rank_window
        Q = params.infection_q

        # Rolling percentile rank for each return horizon
        rank1 = ret1.rolling(W).rank(pct=True)
        rank3 = ret3.rolling(W).rank(pct=True)
        rank5 = ret5.rolling(W).rank(pct=True)

        # A bar is "infected" when all three horizons are simultaneously in bottom Q%
        infected = ((rank1 < Q) & (rank3 < Q) & (rank5 < Q)).astype(float)

        # Epidemic infection rate I(t): rolling fraction of infected bars
        inf_rate = infected.rolling(params.inf_window).mean()

        # Percentile rank of the infection rate — low value signals epidemic clearing
        inf_rank = inf_rate.rolling(W).rank(pct=True)

        # ATR via exponentially-weighted Wilder approximation
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=params.atr_period, adjust=False).mean()

        # Annualized realized vol for inverse-vol position sizing
        vol = ret1.rolling(20).std() * np.sqrt(252)

        ind = pd.DataFrame(index=data.index)
        ind["inf_rank"] = inf_rank
        ind["atr"] = atr
        ind["vol"] = vol
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        closes = data["close"].values
        inf_rank_vals = indicators["inf_rank"].values
        atr_vals = indicators["atr"].values
        vol_vals = indicators["vol"].values

        n = len(closes)
        signal = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        in_trade = False
        entry_price = 0.0
        stop_price = 0.0
        hit_breakeven = False
        prev_ir = np.nan

        for i in range(n):
            c = closes[i]
            a = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
            v_raw = vol_vals[i]
            v = v_raw if (not np.isnan(v_raw) and v_raw > 0.0) else params.target_vol
            ir = inf_rank_vals[i]

            sz = float(np.clip(params.target_vol / v, 0.1, 1.0))

            # Entry: infection rank crosses below the recovery threshold from above
            # (epidemic transitions from active to clearing phase)
            entry_trigger = (
                not np.isnan(ir)
                and not np.isnan(prev_ir)
                and ir < params.entry_pct
                and prev_ir >= params.entry_pct
            )

            if not in_trade:
                if entry_trigger and a > 0.0:
                    in_trade = True
                    entry_price = c
                    stop_price = c - params.atr_stop_mult * a
                    hit_breakeven = False
                    signal[i] = 1
                    size_arr[i] = sz
            else:
                # Once price reaches +breakeven_pct%, lift stop to entry price
                if not hit_breakeven and c >= entry_price * (1.0 + params.breakeven_pct / 100.0):
                    hit_breakeven = True
                    stop_price = max(stop_price, entry_price)
                # Trail stop upward only
                if a > 0.0:
                    trail = c - params.trail_mult * a
                    stop_price = max(stop_price, trail)
                # Exit if price falls to or below the stop
                if c <= stop_price:
                    in_trade = False
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size_arr[i] = sz

            prev_ir = ir

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size_arr

        # Mandatory 1-bar shift: decision at close of bar N, fill at open of bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
