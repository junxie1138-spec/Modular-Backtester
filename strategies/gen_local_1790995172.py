from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    seasonal_window: int = 504
    energy_window: int = 16
    energy_ratio: float = 1.2
    seasonal_edge: float = 0.002
    min_phase_samples: int = 30
    profit_target: float = 0.025
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790995172"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.seasonal_window + params.holding_bars + 1,
                       params.energy_window + 8 + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        returns = out["log_return"]
        for scale in (2, 4, 8):
            half = scale // 2
            recent = returns.rolling(half, min_periods=half).sum()
            older = recent.shift(half)
            coefficient = (recent - older) / np.sqrt(float(scale))
            out[f"haar_{scale}"] = coefficient
            out[f"energy_{scale}"] = coefficient.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()

        day = data.index.day
        days_in_month = data.index.days_in_month
        out["phase"] = np.where(day <= 7, 0, np.where(day > days_in_month - 7, 2, 1))
        out["matured_origin_phase"] = out["phase"].shift(params.holding_bars)
        out["matured_return"] = close / close.shift(params.holding_bars) - 1.0
        valid = out["matured_return"].notna() & np.isfinite(out["matured_return"])
        seasonal = pd.Series(np.nan, index=data.index, dtype=float)
        for phase in (0, 1, 2):
            mask = valid & out["matured_origin_phase"].eq(phase)
            count = mask.astype(float).rolling(
                params.seasonal_window, min_periods=params.seasonal_window
            ).sum().shift(1)
            total = out["matured_return"].where(mask, 0.0).rolling(
                params.seasonal_window, min_periods=params.seasonal_window
            ).sum().shift(1)
            mean = total / count.where(count >= params.min_phase_samples)
            out[f"phase_count_{phase}"] = count
            out[f"phase_mean_{phase}"] = mean
            seasonal = seasonal.where(out["phase"].ne(phase), mean)
        out["seasonal_forecast"] = seasonal
        fine_energy = out["energy_2"] + out["energy_4"]
        out["energy_dominance"] = out["energy_8"] / fine_energy.where(fine_energy > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        ready = indicators[["haar_8", "energy_dominance", "seasonal_forecast"]].notna().all(axis=1)
        energetic = indicators["energy_dominance"].ge(params.energy_ratio)
        long_entry = ready & energetic & indicators["haar_8"].gt(0) & indicators["seasonal_forecast"].gt(params.seasonal_edge)
        short_entry = ready & energetic & indicators["haar_8"].lt(0) & indicators["seasonal_forecast"].lt(-params.seasonal_edge)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        candidates[:self.warmup_bars(params)] = 0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and np.isfinite(closes[i]) else np.nan
                if not valid_entry or gain >= params.profit_target or age >= params.holding_bars:
                    position = 0
                    entry_bar = -1
                    entry_price = np.nan
                    signals[i] = 0
                    continue
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
