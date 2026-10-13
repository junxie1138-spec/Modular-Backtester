from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ContagionParams:
    return_rank_window: int = 63
    recurrence_window: int = 32
    contagion_rank_window: int = 126
    entry_percentile: float = 0.80
    stop_atr: float = 2.5
    holding_bars: int = 18
    atr_window: int = 14
    trend_window: int = 200


class GeneratedStrategy(BaseStrategy[ContagionParams]):
    strategy_id = "gen_local-w2_1790993639"

    @classmethod
    def params_type(cls) -> type[ContagionParams]:
        return ContagionParams

    def warmup_bars(self, params: ContagionParams) -> int:
        return int(max(
            params.trend_window,
            params.atr_window + 1,
            params.return_rank_window + params.recurrence_window
            + params.contagion_rank_window + 2,
        ))

    def indicators(self, data: pd.DataFrame, params: ContagionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_rank"] = out["return"].rolling(
            params.return_rank_window, min_periods=params.return_rank_window
        ).rank(pct=True)
        out["trend_ma"] = close.rolling(
            params.trend_window, min_periods=params.trend_window
        ).mean()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        valid = out["return_rank"].notna()
        for name, event in (
            ("up", out["return_rank"] >= 0.80),
            ("down", out["return_rank"] <= 0.20),
        ):
            out[name + "_event"] = event.astype(float).where(valid)
            out[name + "_previous_event"] = out[name + "_event"].shift(1)
            out[name + "_pair"] = out[name + "_event"] * out[name + "_previous_event"]
            out[name + "_frequency"] = out[name + "_event"].rolling(
                params.recurrence_window, min_periods=params.recurrence_window
            ).mean()
            out[name + "_opportunities"] = out[name + "_previous_event"].rolling(
                params.recurrence_window, min_periods=params.recurrence_window
            ).sum()
            out[name + "_pairs"] = out[name + "_pair"].rolling(
                params.recurrence_window, min_periods=params.recurrence_window
            ).sum()
            denominator = out[name + "_opportunities"].where(out[name + "_opportunities"] >= 4)
            out[name + "_conditional"] = out[name + "_pairs"] / denominator
            out[name + "_contagion"] = out[name + "_conditional"] - out[name + "_frequency"]
            out[name + "_contagion_rank"] = out[name + "_contagion"].rolling(
                params.contagion_rank_window, min_periods=params.contagion_rank_window
            ).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ContagionParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        trend = indicators["trend_ma"].to_numpy(dtype=float)
        ranks = indicators["return_rank"]
        long_entry = (
            (data["close"] > indicators["trend_ma"])
            & (indicators["up_contagion"] > 0)
            & (indicators["up_contagion_rank"] >= params.entry_percentile)
            & (ranks >= 0.50)
        ).fillna(False).to_numpy(dtype=bool)
        short_entry = (
            (data["close"] < indicators["trend_ma"])
            & (indicators["down_contagion"] > 0)
            & (indicators["down_contagion_rank"] >= params.entry_percentile)
            & (ranks <= 0.50)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        frozen_atr = np.nan
        stop_price = np.nan
        first_decision = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if side != 0:
                if i == entry_bar:
                    entry_price = opening[i]
                    stop_price = entry_price - side * params.stop_atr * frozen_atr
                valid_position = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(close[i]) and close[i] > 0
                    and np.isfinite(trend[i]) and np.isfinite(stop_price)
                )
                stopped = valid_position and (
                    (side == 1 and close[i] < stop_price)
                    or (side == -1 and close[i] > stop_price)
                )
                regime_exit = valid_position and (
                    (side == 1 and close[i] <= trend[i])
                    or (side == -1 and close[i] >= trend[i])
                )
                expired = i - entry_bar + 1 >= params.holding_bars
                if not valid_position or stopped or regime_exit or expired:
                    side = 0
                    signals[i] = 0
                    continue
                signals[i] = side
                continue
            if i < first_decision:
                continue
            if not (np.isfinite(close[i]) and close[i] > 0
                    and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(trend[i])):
                continue
            if long_entry[i]:
                side = 1
            elif short_entry[i]:
                side = -1
            if side != 0:
                entry_bar = i + 1
                entry_price = np.nan
                frozen_atr = atr[i]
                stop_price = np.nan
                signals[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
