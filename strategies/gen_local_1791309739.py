from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EnergyPositionParams:
    lookback: int = 9
    dominance: float = 0.70


class GeneratedStrategy(BaseStrategy[EnergyPositionParams]):
    strategy_id = "gen_local_1791309739"

    @classmethod
    def params_type(cls) -> type[EnergyPositionParams]:
        return EnergyPositionParams

    @staticmethod
    def warmup_bars(params: EnergyPositionParams) -> int:
        if not 3 <= params.lookback <= 9:
            raise ValueError("lookback must be between 3 and 9")
        if not 0.5 < params.dominance < 1.0:
            raise ValueError("dominance must be between 0.5 and 1.0")
        return int(params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: EnergyPositionParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        upside = returns.clip(lower=0.0).pow(2)
        downside = returns.clip(upper=0.0).pow(2)
        up_window = upside.rolling(params.lookback, min_periods=params.lookback)
        down_window = downside.rolling(params.lookback, min_periods=params.lookback)
        trimmed_up = (up_window.sum() - up_window.max()).clip(lower=0.0)
        trimmed_down = (down_window.sum() - down_window.max()).clip(lower=0.0)
        total = trimmed_up + trimmed_down
        position = (trimmed_up / total.replace(0.0, np.nan)).clip(0.0, 1.0)
        return pd.DataFrame({
            "return": returns.fillna(0.0),
            "upside_energy": upside.fillna(0.0),
            "downside_energy": downside.fillna(0.0),
            "trimmed_upside": trimmed_up.fillna(0.0),
            "trimmed_downside": trimmed_down.fillna(0.0),
            "total_energy": total.fillna(0.0),
            "energy_position": position.fillna(0.5),
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EnergyPositionParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        position = indicators["energy_position"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        # Neutral observations retain the regime; only opposite dominance exits.
        events = pd.Series(np.select(
            [position >= params.dominance, position <= 1.0 - params.dominance],
            [1.0, 0.0],
            default=np.nan,
        ), index=data.index)
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0.0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = EnergyPositionParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    flat = GeneratedStrategy.indicators(data, params)
    assert GeneratedStrategy.warmup_bars(params) == 10
    assert flat.notna().all().all()
    assert flat["energy_position"].eq(0.5).all()
    synthetic = pd.DataFrame({
        "energy_position": [np.nan, 0.8, 0.5, 0.2, 0.5, 0.8, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, synthetic, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    closes = pd.Series([100, 101, 103, 102, 104, 105, 104, 106, 107, 109, 108, 110], index=index)
    data["close"] = closes
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:10], params)
    pd.testing.assert_frame_equal(full.iloc[:10], prefix)
    assert full["energy_position"].iloc[:9].eq(0.5).all()


if __name__ == "__main__":
    _self_check()
