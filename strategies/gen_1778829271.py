from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_lookback: int = 20
    vol_window: int = 20
    vol_rank_window: int = 252
    regime_threshold: float = 0.5
    min_dd: float = 0.02
    saturation_dd: float = 0.10
    size_floor: float = 0.20
    size_cap: float = 1.00


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_1778829271"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(params.vol_rank_window + params.vol_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)

        # Drawdown depth from rolling local high (>= 0)
        rolling_max = close.rolling(int(params.dd_lookback), min_periods=1).max()
        dd = ((rolling_max - close) / rolling_max).clip(lower=0.0).fillna(0.0)

        # Realized vol of log returns and its trailing percentile rank (regime indicator)
        log_ret = np.log(close / close.shift(1))
        rv = log_ret.rolling(int(params.vol_window), min_periods=int(params.vol_window)).std()
        vol_rank = rv.rolling(int(params.vol_rank_window), min_periods=int(params.vol_rank_window)).rank(pct=True)

        # Elastic regime = low realized-vol percentile (below yield threshold)
        is_elastic_raw = vol_rank < float(params.regime_threshold)
        is_elastic = is_elastic_raw.fillna(False).astype(bool)

        # Trigger: drawdown depth crosses min_dd from below, and regime label is valid
        in_dd = (dd >= float(params.min_dd)).fillna(False).astype(bool)
        prev_in_dd = in_dd.shift(1).fillna(False).astype(bool)
        valid_regime = vol_rank.notna()
        trigger = in_dd & (~prev_in_dd) & valid_regime

        # Direction at trigger: +1 in elastic regime, -1 in plastic regime
        direction = pd.Series(np.where(is_elastic, 1, -1), index=data.index, dtype=float)

        # Snapshot dd and direction at trigger, carry forward exactly one bar
        # for a 2-bar (post-shift) holding horizon.
        trigger_dd = dd.where(trigger)
        trigger_dir = direction.where(trigger)
        held_dd = trigger_dd.ffill(limit=1)
        held_dir = trigger_dir.ffill(limit=1)

        active = held_dd.notna()

        return pd.DataFrame(
            {
                "dd": dd,
                "vol_rank": vol_rank,
                "is_elastic": is_elastic.astype(float),
                "trigger": trigger.astype(float),
                "held_dd": held_dd,
                "held_dir": held_dir,
                "active": active.astype(float),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        active = indicators["active"] > 0
        held_dir = indicators["held_dir"].fillna(0.0)
        held_dd = indicators["held_dd"].fillna(0.0)

        # Raw signal: direction while active, else 0
        raw_signal = np.where(active.values, held_dir.values, 0.0)
        signal_int = pd.Series(raw_signal, index=data.index).round().astype(int)

        # Size scaled by drawdown depth at the trigger bar (linear ramp, saturated)
        span = max(float(params.saturation_dd) - float(params.min_dd), 1e-9)
        norm = ((held_dd - float(params.min_dd)) / span).clip(0.0, 1.0)
        size_active = float(params.size_floor) + (float(params.size_cap) - float(params.size_floor)) * norm
        size = size_active.where(active, float(params.size_floor))
        size = size.fillna(float(params.size_floor)).clip(lower=1e-3).astype(float)

        df = pd.DataFrame({"signal": signal_int, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
