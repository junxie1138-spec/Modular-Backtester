from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PolarityParams:
    ma_window: int = 20
    polarity_window: int = 8
    discount_z: float = 1.2
    change_z: float = 1.5
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[PolarityParams]):
    strategy_id = "gen_local-w2_1791267195"

    @classmethod
    def params_type(cls) -> type[PolarityParams]:
        return PolarityParams

    def warmup_bars(self, params: PolarityParams) -> int:
        return int(max(params.ma_window, params.polarity_window + 1))

    def indicators(self, data: pd.DataFrame, params: PolarityParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.polarity_window < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not (params.discount_z > 0 and params.change_z > 0 and params.profit_target > 0):
            raise ValueError("Thresholds and profit target must be positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).replace(0, np.nan)
        out["distance_z"] = (close - out["ma"]) / out["price_std"]
        out["intraday_return"] = close / opening - 1.0
        prior = out["intraday_return"].shift(1)
        out["prior_body_mean"] = prior.rolling(params.polarity_window, min_periods=params.polarity_window).mean()
        out["prior_body_std"] = prior.rolling(params.polarity_window, min_periods=params.polarity_window).std(ddof=0).replace(0, np.nan)
        out["polarity_change_z"] = (out["intraday_return"] - out["prior_body_mean"]) / out["prior_body_std"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PolarityParams) -> SignalFrame:
        discount = indicators["distance_z"] <= -params.discount_z
        polarity_switch = (
            (indicators["prior_body_mean"] < 0)
            & (indicators["intraday_return"] > 0)
            & (indicators["polarity_change_z"] >= params.change_z)
        )
        entries = (discount & polarity_switch).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                if entry_bar == -1:
                    entry_price = opens[i]
                    entry_bar = i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                target_hit = valid_entry and valid_close and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if not valid_entry or target_hit or time_up:
                    active = False
                    entry_price = np.nan
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
                active = True
                entry_bar = -1
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
