from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalCapacityParams:
    seasonal_window: int = 504
    holding_bars: int = 8
    volume_window: int = 20
    atr_window: int = 14
    min_phase_samples: int = 18
    volume_multiple: float = 1.25
    impulse_threshold: float = 0.004
    seasonal_threshold: float = 0.003
    breakeven_gain: float = 0.012
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalCapacityParams]):
    strategy_id = "gen_local-w2_1790985288"

    @classmethod
    def params_type(cls) -> type[SeasonalCapacityParams]:
        return SeasonalCapacityParams

    def warmup_bars(self, params: SeasonalCapacityParams) -> int:
        return int(max(
            params.seasonal_window + params.holding_bars + 1,
            params.volume_window + 1,
            params.atr_window + 1,
            3,
        ))

    def indicators(self, data: pd.DataFrame, params: SeasonalCapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["impulse"] = close.pct_change(2, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volume_baseline"] = volume.shift(1).rolling(params.volume_window, min_periods=params.volume_window).mean()
        out["volume_ratio"] = volume.div(out["volume_baseline"].where(out["volume_baseline"] > 0)).replace([np.inf, -np.inf], np.nan)

        # Six calendar phases: days 1-5, 6-10, 11-15, 16-20, 21-25, 26-31.
        out["phase"] = pd.Series(np.minimum((data.index.day - 1) // 5, 5), index=data.index, dtype=int)
        out["origin_phase"] = out["phase"].shift(params.holding_bars)
        out["completed_return"] = close.pct_change(params.holding_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["seasonal_forecast"] = np.nan
        for phase in range(6):
            valid = out["origin_phase"].eq(phase) & out["completed_return"].notna()
            # Only returns completed before the current decision bar contribute.
            observations = valid.astype(float).shift(1)
            outcomes = out["completed_return"].where(valid, 0.0).shift(1)
            count = observations.rolling(params.seasonal_window, min_periods=params.seasonal_window).sum()
            total = outcomes.rolling(params.seasonal_window, min_periods=params.seasonal_window).sum()
            forecast = total.div(count.where(count >= params.min_phase_samples))
            out[f"phase_{phase}_count"] = count
            out[f"phase_{phase}_forecast"] = forecast
            current_phase = out["phase"].eq(phase)
            out.loc[current_phase, "seasonal_forecast"] = forecast.loc[current_phase]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalCapacityParams) -> SignalFrame:
        seasonal = indicators["seasonal_forecast"]
        impulse = indicators["impulse"]
        volume_ok = indicators["volume_ratio"].ge(params.volume_multiple)
        long_entry = seasonal.gt(params.seasonal_threshold) & impulse.gt(params.impulse_threshold) & volume_ok
        short_entry = seasonal.lt(-params.seasonal_threshold) & impulse.lt(-params.impulse_threshold) & volume_ok
        candidates = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0).astype(int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        pending_entry = False
        entry_price = 0.0
        entry_atr = 0.0
        stop = 0.0
        armed = False
        age = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if pending_entry:
                # The previous close's decision fills at this bar's open.
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    direction = 0
                    pending_entry = False
                    continue
                entry_price = opens[i]
                # Signed-price coordinates make the stop increase for either direction.
                stop = direction * entry_price - params.trail_atr * entry_atr
                armed = False
                age = 0
                pending_entry = False

            if direction != 0:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    decisions[i] = direction
                    continue
                signed_close = direction * closes[i]
                exit_now = signed_close <= stop or age >= params.holding_bars
                if not exit_now:
                    gain = direction * (closes[i] / entry_price - 1.0)
                    if gain >= params.breakeven_gain:
                        armed = True
                    if armed:
                        stop = max(stop, direction * entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, signed_close - params.trail_atr * atrs[i])
                    exit_now = signed_close <= stop
                if exit_now:
                    direction = 0
                    # Exit decisions are executed on the next bar after shifting.
                    continue
                decisions[i] = direction
                continue

            if i >= warmup - 1 and candidates[i] != 0 and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(candidates[i])
                entry_atr = atrs[i]
                pending_entry = True
                decisions[i] = direction

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
