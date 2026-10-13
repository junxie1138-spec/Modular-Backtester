from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    pattern_window: int = 12
    history_bars: int = 252
    forecast_days: int = 4
    max_distance: float = 0.30
    min_analogs: int = 5
    edge_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791209835"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(200, params.history_bars + params.pattern_window + 3))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (params.pattern_window >= 3 and params.history_bars > params.forecast_days >= 1):
            raise ValueError("Invalid ordinal window or analog horizon")
        if not (0 < params.max_distance <= 1 and params.min_analogs >= 1 and params.edge_threshold > 0):
            raise ValueError("Invalid analog selection parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        # Stable ordering breaks equal-return ties by their time position.
        code = (a > b).astype(int) + 2 * (a > c).astype(int) + 4 * (b > c).astype(int)
        codes = (0, 1, 3, 4, 6, 7)
        columns = []
        for value in codes:
            name = "pattern_" + str(value)
            columns.append(name)
            out[name] = code.eq(value).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
        probabilities = out[columns]
        entropy_terms = probabilities * np.log(probabilities.clip(lower=1e-12))
        out["complexity"] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["volatility"] = returns.rolling(
            params.pattern_window, min_periods=params.pattern_window
        ).std(ddof=0)
        weighted_return = pd.Series(0.0, index=data.index)
        total_weight = pd.Series(0.0, index=data.index)
        count = pd.Series(0, index=data.index, dtype=int)
        # Sample non-overlapping forward outcomes; every outcome is already known.
        for lag in range(params.forecast_days + 1, params.history_bars + 1, params.forecast_days):
            distance = (probabilities - probabilities.shift(lag)).abs().sum(axis=1, min_count=6) / 2.0
            complexity_gap = (out["complexity"] - out["complexity"].shift(lag)).abs()
            historical_return = close.shift(lag - params.forecast_days) / close.shift(lag) - 1.0
            historical_scale = out["volatility"].shift(lag) * np.sqrt(params.forecast_days)
            response = (historical_return / historical_scale.replace(0.0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            ).clip(-3.0, 3.0)
            accepted = distance.le(params.max_distance) & complexity_gap.le(0.10) & response.notna()
            weight = (1.0 - distance / (params.max_distance + 1e-12)).where(accepted, 0.0).fillna(0.0)
            weighted_return += weight * response.fillna(0.0)
            total_weight += weight
            count += accepted.astype(int)
        out["analog_count"] = count
        out["forecast"] = (weighted_return / total_weight.replace(0.0, np.nan)).where(
            count.ge(params.min_analogs)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        forecast = indicators["forecast"]
        evidence = pd.Series(0, index=data.index, dtype=int)
        evidence.loc[forecast.ge(params.edge_threshold)] = 1
        evidence.loc[forecast.le(-params.edge_threshold)] = -1
        ready = indicators[["forecast", "ma200", "complexity"]].notna().all(axis=1)
        regime = pd.Series(0, index=data.index, dtype=int)
        regime.loc[data["close"].gt(indicators["ma200"])] = 1
        regime.loc[data["close"].lt(indicators["ma200"])] = -1
        entries = evidence.where(ready & evidence.eq(regime), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (direction, entry) in enumerate(zip(evidence.to_numpy(), entries.to_numpy())):
            if position != 0:
                if direction == -position:
                    position = 0
            elif entry != 0:
                position = int(entry)
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
