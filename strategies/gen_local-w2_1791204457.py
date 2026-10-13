from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 60
    recovery_bars: int = 18
    origin_extreme: float = 0.8
    recovery_move: float = 1.2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791204457"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(params.ma_window + params.recovery_bars)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.recovery_bars < 1:
            raise ValueError("Lookback windows must be positive; ma_window must be at least two.")
        if not np.isfinite(params.origin_extreme) or params.origin_extreme <= 0:
            raise ValueError("origin_extreme must be finite and positive.")
        if not np.isfinite(params.recovery_move) or params.recovery_move <= 0:
            raise ValueError("recovery_move must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        window = close.rolling(params.ma_window, min_periods=params.ma_window)
        out["ma"] = window.mean()
        out["dispersion"] = window.std(ddof=0)
        out["anchor_ma"] = out["ma"].shift(params.recovery_bars)
        out["anchor_dispersion"] = out["dispersion"].shift(params.recovery_bars).replace(0.0, np.nan)
        out["origin_close"] = close.shift(params.recovery_bars)
        out["origin_z"] = (out["origin_close"] - out["anchor_ma"]) / out["anchor_dispersion"]
        out["current_anchor_z"] = (close - out["anchor_ma"]) / out["anchor_dispersion"]
        out["displacement_z"] = out["current_anchor_z"] - out["origin_z"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        origin = indicators["origin_z"]
        displacement = indicators["displacement_z"]
        valid = np.isfinite(origin) & np.isfinite(displacement)
        entry = valid & (origin <= -params.origin_extreme) & (displacement >= params.recovery_move)
        reversal = valid & (origin >= params.origin_extreme) & (displacement <= -params.recovery_move)
        # Negating both z-scores exchanges entry and exit exactly.
        events = pd.Series(
            np.select([entry, reversal], [1.0, 0.0], default=np.nan),
            index=data.index,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {
            "origin_z": [-2.0, -2.0, 0.0, 2.0, 2.0, -2.0],
            "displacement_z": [0.0, 2.0, 0.0, 0.0, -2.0, 2.0],
        },
        index=index,
    )
    params = RecoveryParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    mirrored = GeneratedStrategy.generate_signals(None, data, -indicators, None, params)
    assert mirrored.data["signal"].tolist() == [0, 0, 0, 0, 0, 1]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
