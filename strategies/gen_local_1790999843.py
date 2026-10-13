from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    block_bars: int = 4
    horizon_bars: int = 4
    history_bars: int = 160
    bandwidth: float = 0.8
    entry_threshold: float = 0.2
    min_effective: float = 6.0
    min_size: float = 0.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790999843"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        block = max(2, int(params.block_bars))
        horizon = max(1, int(params.horizon_bars))
        first_lag = 3 * block + horizon
        history = max(first_lag, int(params.history_bars))
        return int(history + max(20, 3 * block) + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        block = max(2, int(params.block_bars))
        horizon = max(1, int(params.horizon_bars))
        baseline = max(20, 3 * block)
        first_lag = 3 * block + horizon
        history = max(first_lag, int(params.history_bars))
        close = data["close"].astype(float).where(data["close"] > 0)
        returns = close.pct_change(fill_method=None)
        recent_vol = returns.rolling(block, min_periods=block).std(ddof=0)
        middle_vol = recent_vol.shift(block)
        early_vol = recent_vol.shift(2 * block)
        base_vol = returns.rolling(baseline, min_periods=baseline).std(ddof=0)
        recent_vol = recent_vol.where(recent_vol > 1e-12)
        middle_vol = middle_vol.where(middle_vol > 1e-12)
        early_vol = early_vol.where(early_vol > 1e-12)
        base_vol = base_vol.where(base_vol > 1e-12)
        features = pd.DataFrame(index=data.index)
        features["vol_recent_middle"] = np.log(recent_vol / middle_vol)
        features["vol_middle_early"] = np.log(middle_vol / early_vol)
        features["vol_relative_level"] = np.log(recent_vol / base_vol)
        features["displacement"] = (close / close.shift(3 * block) - 1.0) / (base_vol * np.sqrt(3 * block))
        features = features.replace([np.inf, -np.inf], np.nan)
        current = features.to_numpy(dtype=float)
        current_valid = np.isfinite(current).all(axis=1)
        weight_sum = np.zeros(len(data), dtype=float)
        weight_square_sum = np.zeros(len(data), dtype=float)
        outcome_sum = np.zeros(len(data), dtype=float)
        outcome_square_sum = np.zeros(len(data), dtype=float)
        bandwidth = max(0.05, float(params.bandwidth))
        for lag in range(first_lag, history + 1, block):
            historical = features.shift(lag).to_numpy(dtype=float)
            outcome = ((close.shift(lag - horizon) / close.shift(lag) - 1.0) / (base_vol.shift(lag) * np.sqrt(horizon))).to_numpy(dtype=float)
            valid = current_valid & np.isfinite(historical).all(axis=1) & np.isfinite(outcome)
            delta = np.where(valid[:, None], current - historical, 0.0)
            distance = np.sum(delta * delta, axis=1) / current.shape[1]
            weight = np.where(valid, np.exp(-distance / (2.0 * bandwidth * bandwidth)), 0.0)
            safe_outcome = np.where(valid, outcome, 0.0)
            weight_sum += weight
            weight_square_sum += weight * weight
            outcome_sum += weight * safe_outcome
            outcome_square_sum += weight * safe_outcome * safe_outcome
        denominator = np.where(weight_sum > 1e-12, weight_sum, np.nan)
        forecast = outcome_sum / denominator
        variance = np.maximum(outcome_square_sum / denominator - forecast * forecast, 0.0)
        effective = weight_sum * weight_sum / np.where(weight_square_sum > 1e-12, weight_square_sum, np.nan)
        score = forecast / np.sqrt(variance + 0.25)
        out = features.copy()
        out["recent_vol"] = recent_vol
        out["middle_vol"] = middle_vol
        out["early_vol"] = early_vol
        out["base_vol"] = base_vol
        out["forecast"] = forecast
        out["forecast_variance"] = variance
        out["effective_analogs"] = effective
        out["analog_weight"] = weight_sum
        out["score"] = score
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        score = indicators["score"].fillna(0.0).to_numpy(dtype=float)
        effective = indicators["effective_analogs"].fillna(0.0).to_numpy(dtype=float)
        mass = indicators["analog_weight"].fillna(0.0).to_numpy(dtype=float)
        threshold = max(0.01, float(params.entry_threshold))
        eligible = (effective >= max(1.0, float(params.min_effective))) & (mass >= 1.0)
        eligible[:self.warmup_bars(params)] = False
        desired = np.where(eligible & (score >= threshold), 1, np.where(eligible & (score <= -threshold), -1, 0))
        lower_size = max(0.001, min(float(params.min_size), 1.0))
        upper_size = max(lower_size, min(float(params.max_size), 1.0))
        conviction = np.clip((np.abs(score) - threshold) / (3.0 * threshold), 0.0, 1.0)
        proposed_size = lower_size + (upper_size - lower_size) * conviction
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), lower_size, dtype=float)
        position = 0
        entry_size = lower_size
        for i in range(len(data)):
            direction = int(desired[i])
            if position != 0 and direction == -position:
                position = 0
                entry_size = lower_size
            elif position == 0 and direction != 0:
                position = direction
                entry_size = float(proposed_size[i])
            signals[i] = position
            sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(lower_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
