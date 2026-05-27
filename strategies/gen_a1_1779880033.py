from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    momentum_lookback: int = 5
    pct_window: int = 252
    pct_threshold: float = 0.70
    er_window: int = 10
    er_pct_threshold: float = 0.65
    min_streak: int = 3
    trend_ma: int = 200
    atr_window: int = 14
    initial_stop_atr: float = 2.5
    breakeven_trigger: float = 0.025
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy["GeneratedParams"]):
    strategy_id = "gen_a1_1779880033"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return int(max(
            params.trend_ma,
            params.pct_window + params.momentum_lookback + 1,
            params.er_window + params.pct_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        mom = close.pct_change(params.momentum_lookback)
        min_p = max(20, params.pct_window // 4)
        pct_thresh = mom.rolling(params.pct_window, min_periods=min_p).quantile(params.pct_threshold)
        thrust = (mom > pct_thresh).fillna(False).astype(int)

        thrust_shift = thrust.shift(fill_value=0)
        group_id = (thrust != thrust_shift).cumsum()
        within_group = thrust.groupby(group_id).cumcount() + 1
        streak_len = within_group.where(thrust == 1, 0).astype(int)

        net_move = (close - close.shift(params.er_window)).abs()
        path = close.diff().abs().rolling(params.er_window).sum()
        er = (net_move / path.replace(0, np.nan)).clip(0.0, 1.0)
        er_pct = er.rolling(params.pct_window, min_periods=min_p).rank(pct=True)
        direction_up = (close > close.shift(params.er_window)).astype(int)

        ma_trend = close.rolling(params.trend_ma).mean()

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        return pd.DataFrame({
            "mom": mom,
            "pct_thresh": pct_thresh,
            "thrust": thrust,
            "streak_len": streak_len,
            "er": er,
            "er_pct": er_pct,
            "direction_up": direction_up,
            "ma_trend": ma_trend,
            "atr": atr,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        streak_len = indicators["streak_len"].to_numpy()
        er_pct = indicators["er_pct"].to_numpy()
        direction_up = indicators["direction_up"].to_numpy()
        ma_trend = indicators["ma_trend"].to_numpy()
        atr = indicators["atr"].to_numpy()

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)

        in_position = False
        entry_price = 0.0
        stop = 0.0
        in_trade_high = 0.0
        breakeven_hit = False

        for i in range(n):
            c = close[i]
            a = atr[i]
            mt = ma_trend[i]
            ep = er_pct[i]
            sl = streak_len[i]
            du = direction_up[i]

            if not in_position:
                ok = (
                    not np.isnan(c)
                    and not np.isnan(mt)
                    and not np.isnan(a)
                    and not np.isnan(ep)
                    and sl >= params.min_streak
                    and ep >= params.er_pct_threshold
                    and du == 1
                    and c > mt
                    and a > 0.0
                )
                if ok:
                    in_position = True
                    entry_price = float(c)
                    stop = float(c) - params.initial_stop_atr * float(a)
                    in_trade_high = float(c)
                    breakeven_hit = False
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                if c > in_trade_high:
                    in_trade_high = float(c)
                if not breakeven_hit and c >= entry_price * (1.0 + params.breakeven_trigger):
                    if entry_price > stop:
                        stop = entry_price
                    breakeven_hit = True
                if breakeven_hit and not np.isnan(a):
                    new_stop = in_trade_high - params.trail_atr * float(a)
                    if new_stop > stop:
                        stop = new_stop
                if c <= stop:
                    in_position = False
                    raw[i] = 0
                    entry_price = 0.0
                    stop = 0.0
                    in_trade_high = 0.0
                    breakeven_hit = False
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        signal = pd.Series(raw, index=data.index, dtype=np.int64)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
