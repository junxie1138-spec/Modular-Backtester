from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PreyRecoveryParams:
    prey_window: int = 20
    pred_window: int = 20
    rank_window: int = 60
    regime_window: int = 200
    prey_low_thresh: float = 0.40
    pred_high_thresh: float = 0.60
    prey_velocity_bars: int = 3
    pred_velocity_bars: int = 3
    profit_target_pct: float = 0.02
    time_stop_bars: int = 5


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779648034"

    @classmethod
    def params_type(cls):
        return PreyRecoveryParams

    @staticmethod
    def warmup_bars(params: PreyRecoveryParams) -> int:
        sub = (
            params.rank_window
            + max(params.prey_window, params.pred_window)
            + max(params.prey_velocity_bars, params.pred_velocity_bars)
            + 2
        )
        return int(max(params.regime_window, sub))

    def indicators(self, data: pd.DataFrame, params: PreyRecoveryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ind = pd.DataFrame(index=data.index)

        logret = np.log(close).diff()

        prey_raw = logret.rolling(params.prey_window, min_periods=params.prey_window).mean()
        pred_raw = logret.rolling(params.pred_window, min_periods=params.pred_window).std(ddof=0)

        prey_rank = prey_raw.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        pred_rank = pred_raw.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)

        ind["prey_rank"] = prey_rank
        ind["pred_rank"] = pred_rank
        ind["prey_velocity"] = prey_rank - prey_rank.shift(params.prey_velocity_bars)
        ind["pred_velocity"] = pred_rank - pred_rank.shift(params.pred_velocity_bars)

        sma200 = close.rolling(params.regime_window, min_periods=params.regime_window).mean()
        ind["sma200"] = sma200
        ind["regime_up"] = (close > sma200).astype(float)
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PreyRecoveryParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        prey_rank = indicators["prey_rank"].to_numpy(dtype=float)
        pred_rank = indicators["pred_rank"].to_numpy(dtype=float)
        prey_vel = indicators["prey_velocity"].to_numpy(dtype=float)
        pred_vel = indicators["pred_velocity"].to_numpy(dtype=float)
        regime = indicators["regime_up"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_price = 0.0
        entry_bar = -1

        prey_low = float(params.prey_low_thresh)
        pred_high = float(params.pred_high_thresh)
        pt = float(params.profit_target_pct)
        tstop = int(params.time_stop_bars)

        for i in range(n):
            pr = prey_rank[i]
            dr = pred_rank[i]
            pv = prey_vel[i]
            dv = pred_vel[i]
            rg = regime[i]

            if in_pos:
                bars_held = i - entry_bar
                px = close[i]
                if entry_price > 0.0:
                    pnl = (px / entry_price) - 1.0
                else:
                    pnl = 0.0
                if (pnl >= pt) or (bars_held >= tstop):
                    in_pos = False
                    entry_price = 0.0
                    entry_bar = -1
                    raw[i] = 0
                else:
                    raw[i] = 1
                continue

            if (
                np.isnan(pr)
                or np.isnan(dr)
                or np.isnan(pv)
                or np.isnan(dv)
                or np.isnan(rg)
            ):
                raw[i] = 0
                continue

            prey_ready = (pr <= prey_low) and (pv > 0.0)
            pred_ready = (dr >= pred_high) and (dv < 0.0)
            regime_ok = rg > 0.5

            if prey_ready and pred_ready and regime_ok:
                in_pos = True
                entry_price = close[i]
                entry_bar = i
                raw[i] = 1
            else:
                raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
