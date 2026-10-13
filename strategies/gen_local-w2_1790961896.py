from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepricingParams:
    recent_bars: int = 4
    baseline_bars: int = 40
    gap_threshold: float = 2.0
    intraday_threshold: float = 1.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[RepricingParams]):
    strategy_id = "gen_local-w2_1790961896"

    @classmethod
    def params_type(cls) -> type[RepricingParams]:
        return RepricingParams

    def warmup_bars(self, params: RepricingParams) -> int:
        if params.recent_bars < 2 or params.baseline_bars < 2 or params.holding_bars < 1:
            raise ValueError("Windows must be at least two bars; holding_bars must be positive.")
        if not all(np.isfinite(v) and v > 0 for v in (params.gap_threshold, params.intraday_threshold)):
            raise ValueError("Thresholds must be finite and positive.")
        return int(params.recent_bars + params.baseline_bars + 1)

    def indicators(self, data: pd.DataFrame, params: RepricingParams) -> pd.DataFrame:
        self.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        result["gap"] = np.log(opening / closing.shift(1))
        result["intraday"] = np.log(closing / opening)
        scale = np.sqrt(1.0 / params.recent_bars + 1.0 / params.baseline_bars)
        for primitive in ("gap", "intraday"):
            series = result[primitive].replace([np.inf, -np.inf], np.nan)
            result[primitive] = series
            result[primitive + "_recent_mean"] = series.rolling(params.recent_bars, min_periods=params.recent_bars).mean()
            historical = series.shift(params.recent_bars).rolling(params.baseline_bars, min_periods=params.baseline_bars)
            result[primitive + "_baseline_mean"] = historical.mean()
            result[primitive + "_baseline_std"] = historical.std(ddof=1)
            denominator = result[primitive + "_baseline_std"].where(result[primitive + "_baseline_std"] > 1e-12) * scale
            result[primitive + "_shift_score"] = (result[primitive + "_recent_mean"] - result[primitive + "_baseline_mean"]) / denominator
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RepricingParams) -> SignalFrame:
        self.warmup_bars(params)
        gap = indicators["gap_shift_score"]
        intraday = indicators["intraday_shift_score"]
        long_agreement = (gap >= params.gap_threshold) & (intraday >= params.intraday_threshold)
        short_agreement = (gap <= -params.gap_threshold) & (intraday <= -params.intraday_threshold)
        agreement = pd.Series(np.select([long_agreement, short_agreement], [1, -1], default=0), index=data.index, dtype=int)
        entry = agreement.where(agreement.ne(agreement.shift(1).fillna(0)), 0)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        exit_bar = -1
        for bar, candidate in enumerate(entry.to_numpy()):
            if direction != 0:
                if bar == exit_bar:
                    direction = 0
                else:
                    signals[bar] = direction
                continue
            if candidate != 0:
                direction = int(candidate)
                exit_bar = bar + params.holding_bars
                signals[bar] = direction
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RepricingParams()
    strategy = GeneratedStrategy()
    count = 3 * params.holding_bars + 5
    index = pd.date_range("2020-01-01", periods=count, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    scores = pd.DataFrame({"gap_shift_score": 0.0, "intraday_shift_score": 0.0}, index=index)
    scores.iloc[0] = [3.0, 2.0]
    scores.iloc[2] = [-3.0, -2.0]
    scores.iloc[params.holding_bars] = [-3.0, -2.0]
    scores.iloc[params.holding_bars + 2] = [-3.0, -2.0]
    scores.iloc[2 * params.holding_bars + 4] = [3.0, -2.0]
    output = strategy.generate_signals(data, scores, None, params).data
    expected = np.zeros(count, dtype=int)
    expected[1:params.holding_bars + 1] = 1
    expected[params.holding_bars + 3:2 * params.holding_bars + 3] = -1
    assert np.array_equal(output["signal"].to_numpy(), expected)
    flat_indicators = strategy.indicators(data, params)
    assert not strategy.generate_signals(data, flat_indicators, None, params).data["signal"].any()
    assert output["size"].gt(0).all()
    assert strategy.warmup_bars(params) == params.recent_bars + params.baseline_bars + 1
