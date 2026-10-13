from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history_days: int = 90
    volume_window: int = 20
    atr_window: int = 14
    volume_burst: float = 1.3
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791225940"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @classmethod
    def warmup_bars(cls, params: AnalogParams) -> int:
        return int(params.history_days + params.volume_window + 4)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        average_volume = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean()
        out["volume_ratio"] = (data["volume"] / average_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["wave"] = out["return"] * np.sqrt(out["volume_ratio"].clip(lower=0, upper=4))

        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()

        distances = []
        outcomes = []
        for lag in range(3, params.history_days + 3):
            distance = sum((out["wave"].shift(offset) - out["wave"].shift(lag + offset)) ** 2 for offset in range(3))
            distances.append(distance.to_numpy())
            outcomes.append((data["close"].shift(lag - 2) / data["close"].shift(lag) - 1).to_numpy())

        distance_array = np.column_stack(distances)
        outcome_array = np.column_stack(outcomes)
        distance_array = np.where(np.isfinite(distance_array) & np.isfinite(outcome_array), distance_array, np.inf)
        nearest = np.argsort(distance_array, axis=1)[:, :5]
        selected_distances = np.take_along_axis(distance_array, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcome_array, nearest, axis=1)
        valid = np.isfinite(selected_distances).all(axis=1)
        selected_outcomes = np.where(np.isfinite(selected_outcomes), selected_outcomes, 0)
        out["analog_mean"] = np.where(valid, selected_outcomes.mean(axis=1), 0.0)
        out["analog_hit"] = np.where(valid, (selected_outcomes > 0).mean(axis=1), 0.0)
        out["analog_valid"] = valid
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        atr_fraction = (indicators["atr"] / data["close"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        consensus = ((indicators["analog_hit"] - 0.5) * 2).clip(0, 1)
        edge = (indicators["analog_mean"] / atr_fraction.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
        scaled_size = (0.25 + 0.65 * consensus * edge).clip(0.25, 0.90)
        setup = (
            indicators["analog_valid"].astype(bool)
            & (indicators["return"] > 0)
            & (indicators["volume_ratio"] >= params.volume_burst)
            & (indicators["analog_hit"] >= 0.6)
            & (indicators["analog_mean"] > 0)
            & (indicators["atr"] > 0)
        ).to_numpy()

        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        requested_size = scaled_size.to_numpy()
        entry_index = -1
        entry_stop = 0.0
        position_size = 0.25

        for bar in range(len(data)):
            held = bar > 0 and signals[bar - 1] == 1
            if held:
                if entry_index < 0:
                    entry_index = bar
                    entry_stop = opens[bar] - params.stop_atr * atr[bar - 1]
                    position_size = sizes[bar - 1]
                if closes[bar] < entry_stop or bar - entry_index >= 1:
                    signals[bar] = 0
                    entry_index = -1
                else:
                    signals[bar] = 1
                    sizes[bar] = position_size
            elif setup[bar]:
                signals[bar] = 1
                sizes[bar] = float(requested_size[bar])

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
