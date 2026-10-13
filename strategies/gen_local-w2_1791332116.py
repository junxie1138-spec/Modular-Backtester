from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_length: int = 3
    regime_window: int = 20


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791332116"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.regime_window, params.streak_length) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.streak_length < 1 or params.regime_window < 3:
            raise ValueError("streak_length must be positive and regime_window at least 3")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["absolute_return"] = out["return"].abs()
        out["volume_change"] = data["volume"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["return"]).fillna(0).astype(int)
        out["run_group"] = out["direction"].ne(out["direction"].shift()).cumsum()
        out["streak"] = (out.groupby("run_group").cumcount() + 1).where(out["direction"].ne(0), 0)
        out["coupling"] = out["absolute_return"].rolling(params.regime_window, min_periods=params.regime_window).corr(out["volume_change"]).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        eligible = indicators["streak"].eq(params.streak_length) & indicators["coupling"].abs().ge(0.2)
        candidates = (indicators["direction"] * np.sign(indicators["coupling"])).where(eligible, 0).fillna(0).astype(int).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position and i == exit_bar:
                position = 0
            elif not position and candidate:
                position = int(candidate)
                exit_bar = i + 4
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "streak": [3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
        "direction": [1, 0, -1, 0, -1, 0, -1, 0, 1, 0, 1, 0],
        "coupling": [0.5] * 12,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0]
    indicators["coupling"] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params()).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
