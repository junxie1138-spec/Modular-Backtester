from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BarrierFatigueParams:
    channel_window: int = 15
    test_window: int = 8
    rank_window: int = 60
    atr_window: int = 14
    test_distance_atr: float = 0.35
    fatigue_percentile: float = 0.80
    breakeven_pct: float = 0.008
    trail_atr: float = 1.8
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[BarrierFatigueParams]):
    strategy_id = "gen_local_1790989799"

    @classmethod
    def params_type(cls) -> type[BarrierFatigueParams]:
        return BarrierFatigueParams

    def warmup_bars(self, params: BarrierFatigueParams) -> int:
        return int(max(params.channel_window, params.atr_window + 1) + params.test_window + params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: BarrierFatigueParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["ceiling"] = data["high"].rolling(params.channel_window, min_periods=params.channel_window).max().shift(1)
        out["floor"] = data["low"].rolling(params.channel_window, min_periods=params.channel_window).min().shift(1)
        out["test_margin"] = params.test_distance_atr * out["atr"].shift(1)
        valid = out[["ceiling", "floor", "test_margin"]].notna().all(axis=1)
        out["upper_test"] = ((data["high"] >= out["ceiling"] - out["test_margin"]) & (close <= out["ceiling"])).astype(float).where(valid)
        out["lower_test"] = ((data["low"] <= out["floor"] + out["test_margin"]) & (close >= out["floor"])).astype(float).where(valid)
        out["upper_density"] = out["upper_test"].rolling(params.test_window, min_periods=params.test_window).mean()
        out["lower_density"] = out["lower_test"].rolling(params.test_window, min_periods=params.test_window).mean()
        out["upper_rank"] = out["upper_density"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["lower_rank"] = out["lower_density"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["first_long"] = ((close > out["ceiling"]) & (previous_close <= out["ceiling"]) & (out["upper_rank"].shift(1) >= params.fatigue_percentile)).fillna(False)
        out["first_short"] = ((close < out["floor"]) & (previous_close >= out["floor"]) & (out["lower_rank"].shift(1) >= params.fatigue_percentile)).fillna(False)
        out["confirmed_long"] = (out["first_long"].shift(1, fill_value=False) & (close > out["ceiling"].shift(1)) & (close > previous_close)).fillna(False)
        out["confirmed_short"] = (out["first_short"].shift(1, fill_value=False) & (close < out["floor"].shift(1)) & (close < previous_close)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BarrierFatigueParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["confirmed_long"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["confirmed_short"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        direction = 0
        entry = np.nan
        signed_stop = -np.inf
        armed = False
        held = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            price = closes[i]
            atr = atrs[i]
            valid = np.isfinite(price) and price > 0.0 and np.isfinite(atr) and atr > 0.0
            if not valid:
                direction = 0
                entry = np.nan
                armed = False
                held = 0
                continue

            if direction != 0:
                # The preceding close's decision fills at this bar's open.
                # This open affects only this bar's subsequent close decision.
                if not np.isfinite(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0.0:
                        direction = 0
                        entry = np.nan
                        continue
                    entry_atr = atrs[i - 1]
                    signed_stop = direction * entry - params.trail_atr * entry_atr
                    armed = False
                    held = 0

                held += 1
                signed_price = direction * price
                # Stops are evaluated at closes; no intrabar ordering is assumed.
                if signed_price <= signed_stop or held >= params.max_hold_bars:
                    direction = 0
                    entry = np.nan
                    armed = False
                    held = 0
                    continue

                profit = direction * (price / entry - 1.0)
                if profit >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Direction-normalized stops only rise. For shorts this
                    # corresponds to an actual stop price that only falls.
                    signed_stop = max(signed_stop, direction * entry, signed_price - params.trail_atr * atr)
                signals[i] = direction
                continue

            if i + 1 < warmup:
                continue
            if longs[i] and not shorts[i]:
                direction = 1
            elif shorts[i] and not longs[i]:
                direction = -1
            else:
                continue
            entry = np.nan
            signed_stop = -np.inf
            armed = False
            held = 0
            signals[i] = direction

        df = pd.DataFrame({"signal": signals, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
