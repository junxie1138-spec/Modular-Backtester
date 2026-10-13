from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeResponseParams:
    lookback: int = 25
    volume_multiple: float = 0.9
    move_multiple: float = 0.4


class GeneratedStrategy(BaseStrategy[VolumeResponseParams]):
    strategy_id = "gen_local_1791300038"

    @classmethod
    def params_type(cls) -> type[VolumeResponseParams]:
        return VolumeResponseParams

    @classmethod
    def warmup_bars(cls, params: VolumeResponseParams) -> int:
        return 2 * params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: VolumeResponseParams) -> pd.DataFrame:
        n = params.lookback
        ret = data["close"].pct_change(fill_method=None)
        volume_mean = data["volume"].rolling(n, min_periods=n).mean().shift(1)
        move_mean = ret.abs().rolling(n, min_periods=n).mean().shift(1)
        relative_volume = data["volume"].div(volume_mean.replace(0, np.nan))
        confirmed_move = (
            relative_volume.ge(params.volume_multiple)
            & ret.abs().ge(params.move_multiple * move_mean)
            & ret.ne(0)
        )
        impulse = np.sign(ret).where(confirmed_move, 0).fillna(0).astype(int)
        response = impulse.shift(1).fillna(0) * ret.fillna(0)
        response_score = response.rolling(n, min_periods=n).sum()
        regime = np.sign(response_score).fillna(0).astype(int)
        ready = pd.Series(np.arange(len(data)) >= self.warmup_bars(params), index=data.index)
        candidate = (impulse * regime).where(ready, 0).astype(int)

        return pd.DataFrame({
            "return": ret.fillna(0),
            "relative_volume": relative_volume.fillna(0),
            "move_mean": move_mean.fillna(0),
            "impulse": impulse,
            "response_score": response_score.fillna(0),
            "regime": regime,
            "candidate": candidate,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolumeResponseParams,
    ) -> SignalFrame:
        candidate = indicators["candidate"]
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).fillna(0).astype(int)
        position = 0
        signals = []
        for direction in confirmed:
            if position == 0 and direction != 0:
                position = int(direction)
            elif position != 0 and direction == -position:
                position = 0
            signals.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index, dtype=int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
