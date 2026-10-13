from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 4
    baseline_bars: int = 16
    trigger: float = 1.0
    efficiency_min: float = 0.25


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791194206"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(2 * params.path_bars + params.baseline_bars)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.path_bars < 2 or params.baseline_bars < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not np.isfinite(params.trigger) or params.trigger <= 0:
            raise ValueError("trigger must be finite and positive.")
        if not np.isfinite(params.efficiency_min) or not 0 < params.efficiency_min <= 1:
            raise ValueError("efficiency_min must be in (0, 1].")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap_magnitude"] = np.log(opening / closing.shift(1)).abs()
        out["body_return"] = np.log(closing / opening)
        out = out.replace([np.inf, -np.inf], np.nan)
        x = pd.Series(0.0, index=data.index)
        y = x.copy()
        absolute_y = x.copy()
        area = x.copy()
        bound = x.copy()
        # Antisymmetric level-two signature of cumulative |gap| and body returns.
        for lag in range(params.path_bars - 1, -1, -1):
            dx = out["gap_magnitude"].shift(lag)
            dy = out["body_return"].shift(lag)
            area = area + x * dy - y * dx
            bound = bound + x * dy.abs() + absolute_y * dx
            x = x + dx
            y = y + dy
            absolute_y = absolute_y + dy.abs()
        out["signature_area"] = area / bound.clip(lower=1e-12)
        out["body_net"] = y
        out["body_activity"] = absolute_y
        out["efficiency"] = y / absolute_y.clip(lower=1e-12)
        historical_area = out["signature_area"].shift(params.path_bars)
        baseline = historical_area.rolling(params.baseline_bars, min_periods=params.baseline_bars)
        out["area_baseline"] = baseline.mean()
        out["area_noise"] = baseline.std(ddof=0).clip(lower=0.1)
        out["change_score"] = (out["signature_area"] - out["area_baseline"]) / out["area_noise"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        score = indicators["change_score"]
        efficiency = indicators["efficiency"]
        long_entry = (score >= params.trigger) & (efficiency >= params.efficiency_min)
        short_entry = (score <= -params.trigger) & (efficiency <= -params.efficiency_min)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # An opposing entry closes first; re-entry requires a later bar.
                position = 0
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame(
        {"change_score": [0, 2, 0, -2, -2, 0, 2, 2],
         "efficiency": [0, 1, 0, -1, -1, 0, 1, 1]},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    reflected = strategy.generate_signals(data, -features, ctx, params).data
    assert reflected["signal"].equals(-result["signal"])
    assert result["size"].gt(0).all()
