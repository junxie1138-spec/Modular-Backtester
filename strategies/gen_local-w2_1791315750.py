from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEchoParams:
    lookback: int = 90
    profit_target: float = 0.04


class GeneratedStrategy(BaseStrategy[GapEchoParams]):
    strategy_id = "gen_local-w2_1791315750"

    @classmethod
    def params_type(cls) -> type[GapEchoParams]:
        return GapEchoParams

    @staticmethod
    def warmup_bars(params: GapEchoParams) -> int:
        return max(20, int(params.lookback)) + 11

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapEchoParams) -> pd.DataFrame:
        window = max(20, int(params.lookback))
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        gap = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        lag5 = gap.shift(5)
        lag10 = gap.shift(10)
        mean = gap.rolling(window, min_periods=window).mean()
        scale = gap.rolling(window, min_periods=window).std(ddof=0)
        rho5 = gap.rolling(window, min_periods=window).corr(lag5)
        rho10 = gap.rolling(window, min_periods=window).corr(lag10)
        z = (gap - mean) / scale.where(scale > 0)
        return pd.DataFrame({
            "gap": gap,
            "gap_lag5": lag5,
            "gap_lag10": lag10,
            "gap_mean": mean,
            "gap_scale": scale,
            "rho5": rho5,
            "rho10": rho10,
            "gap_z": z,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEchoParams,
    ) -> SignalFrame:
        ready = indicators[["rho5", "rho10", "gap_z"]].notna().all(axis=1)
        regime = ready & indicators["rho5"].lt(-0.20) & indicators["rho10"].gt(0.20)
        candidates = np.zeros(len(data), dtype=int)
        candidates[(regime & indicators["gap_z"].lt(-1.0)).to_numpy()] = 1
        candidates[(regime & indicators["gap_z"].gt(1.0)).to_numpy()] = -1
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        target = max(0.001, float(params.profit_target))

        for i in range(len(data)):
            if direction != 0:
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        direction = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = direction * (closes[i] / entry_price - 1.0)
                timed_out = i - entry_bar + 1 >= 20
                if timed_out or (np.isfinite(gain) and gain >= target):
                    direction = 0
                    entry_price = np.nan
                    entry_bar = -1
                raw[i] = direction
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(candidates[i])
                entry_bar = -1
                raw[i] = direction

        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"rho5": -0.5, "rho10": 0.5,
                               "gap_z": 0.0}, index=index)
    indicators.iloc[0, indicators.columns.get_loc("gap_z")] = -2.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapEchoParams()).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:21].eq(1).all()
    assert result["signal"].iloc[21:].eq(0).all()
    for gap_z, direction, target_close in [(-2.0, 1, 105.0), (2.0, -1, 95.0)]:
        indicators.iloc[0, indicators.columns.get_loc("gap_z")] = gap_z
        data["close"] = 100.0
        data.iloc[3, data.columns.get_loc("close")] = target_close
        result = GeneratedStrategy.generate_signals(data, indicators, None, GapEchoParams()).data
        assert result["signal"].iloc[1:4].eq(direction).all()
        assert result["signal"].iloc[4:].eq(0).all()
    derived = GeneratedStrategy.indicators(data, GapEchoParams())
    result = GeneratedStrategy.generate_signals(data, derived, None, GapEchoParams()).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
