from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    debt_bars: int = 6
    volatility_bars: int = 20
    turnover_bars: int = 10
    min_stretch: float = 1.2
    min_repayment: float = 0.15
    max_repayment: float = 0.65
    exhaustion_ratio: float = 0.55
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791286135"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.debt_bars + 2, params.volatility_bars + 2,
                       params.turnover_bars + 3))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.debt_bars, params.volatility_bars, params.turnover_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (3 <= params.hold_bars <= 5):
            raise ValueError("hold_bars must be between three and five.")
        if not (params.min_stretch > 0 and
                0 < params.min_repayment < params.max_repayment < 1 and
                0 < params.exhaustion_ratio < 1):
            raise ValueError("Invalid recovery thresholds.")
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["debt"] = out["return"].rolling(params.debt_bars).sum().shift(1)
        out["volatility"] = out["return"].rolling(params.volatility_bars).std().shift(1)
        scale = out["volatility"].where(out["volatility"] > 0) * np.sqrt(params.debt_bars)
        out["stretch"] = out["debt"] / scale
        out["repayment"] = -out["return"] / out["debt"].where(out["debt"] != 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["sell_volume"] = volume.where(out["return"] < 0, 0.0).where(out["return"].notna())
        out["buy_volume"] = volume.where(out["return"] > 0, 0.0).where(out["return"].notna())
        for side in ("sell", "buy"):
            out[side + "_recent"] = out[side + "_volume"].rolling(2).mean()
            out[side + "_baseline"] = out[side + "_volume"].rolling(params.turnover_bars).mean().shift(2)
            baseline = out[side + "_baseline"]
            out[side + "_ratio"] = out[side + "_recent"] / baseline.where(baseline > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        repair = indicators["repayment"].between(params.min_repayment, params.max_repayment)
        long_entry = (repair & (indicators["return"] > 0) &
                      (indicators["stretch"] <= -params.min_stretch) &
                      (indicators["sell_ratio"] <= params.exhaustion_ratio))
        short_entry = (repair & (indicators["return"] < 0) &
                       (indicators["stretch"] >= params.min_stretch) &
                       (indicators["buy_ratio"] <= params.exhaustion_ratio))
        candidates = np.select([long_entry.fillna(False), short_entry.fillna(False)],
                               [1, -1], default=0).astype(int)
        warmup = int(max(params.debt_bars + 2, params.volatility_bars + 2,
                         params.turnover_bars + 3))
        candidates[:warmup] = 0
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position and i == exit_bar:
                position = 0
                continue
            if position == 0 and candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RecoveryParams()
    n = 40
    index = pd.date_range("2020-01-01", periods=n)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({"return": 0.01, "stretch": -2.0,
                             "repayment": 0.3, "sell_ratio": 0.1,
                             "buy_ratio": 0.1}, index=index)
    start = GeneratedStrategy.warmup_bars(None, params)
    features.iloc[start + 1:, features.columns.get_loc("return")] = -0.01
    features.iloc[start + 1:, features.columns.get_loc("stretch")] = 2.0
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].iloc[:start + 1].eq(0).all()
    assert result["signal"].iloc[start + 1:start + 5].eq(1).all()
    assert result["signal"].iloc[start + 5] == 0
    assert result["signal"].iloc[start + 6:start + 10].eq(-1).all()
    assert result["signal"].iloc[start + 10] == 0
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
