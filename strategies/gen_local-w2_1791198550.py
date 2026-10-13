from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EdgeResidenceParams:
    channel_bars: int = 40
    minimum_streak: int = 4
    edge_threshold: float = 0.80
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[EdgeResidenceParams]):
    strategy_id = "gen_local-w2_1791198550"

    @classmethod
    def params_type(cls) -> type[EdgeResidenceParams]:
        return EdgeResidenceParams

    def warmup_bars(self, params: EdgeResidenceParams) -> int:
        return int(params.channel_bars + params.minimum_streak + 2)

    def indicators(self, data: pd.DataFrame, params: EdgeResidenceParams) -> pd.DataFrame:
        if params.channel_bars < 2 or params.minimum_streak < 1:
            raise ValueError("Channel length must be at least two and streak length positive.")
        if not 0.5 < params.edge_threshold < 1.0 or params.holding_bars < 1:
            raise ValueError("Edge threshold must lie between 0.5 and 1; holding period must be positive.")

        out = pd.DataFrame(index=data.index)
        out["channel_high"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max()
        out["channel_low"] = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["valid"] = out["channel_width"].gt(0) & data["close"].notna()
        out["position"] = ((data["close"] - out["channel_low"]) / out["channel_width"].where(out["valid"])).clip(0, 1).fillna(0.5)
        out["position_change"] = out["position"].diff().fillna(0.0)
        out["close_change"] = data["close"].diff().fillna(0.0)
        out["upper_edge"] = out["valid"] & out["position"].ge(params.edge_threshold)
        out["lower_edge"] = out["valid"] & out["position"].le(1.0 - params.edge_threshold)
        for side in ("upper", "lower"):
            edge = out[side + "_edge"]
            out[side + "_streak"] = edge.groupby((~edge).cumsum()).cumsum().astype(int)

        out["inward_up"] = out["valid"] & out["position_change"].gt(0) & out["close_change"].gt(0) & ~out["lower_edge"]
        out["inward_down"] = out["valid"] & out["position_change"].lt(0) & out["close_change"].lt(0) & ~out["upper_edge"]
        out["long_entry"] = (
            out["lower_streak"].shift(2).fillna(0).ge(params.minimum_streak)
            & out["inward_up"]
            & out["inward_up"].shift(1, fill_value=False)
        )
        out["short_entry"] = (
            out["upper_streak"].shift(2).fillna(0).ge(params.minimum_streak)
            & out["inward_down"]
            & out["inward_down"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EdgeResidenceParams,
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("Holding period must be positive.")
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        positions = np.zeros(len(data), dtype=int)
        direction = 0
        exit_bar = -1
        for i in range(len(data)):
            if direction != 0:
                if i == exit_bar:
                    direction = 0
                else:
                    positions[i] = direction
                continue
            if longs[i] != shorts[i]:
                direction = 1 if longs[i] else -1
                exit_bar = i + params.holding_bars
                positions[i] = direction

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = EdgeResidenceParams()
    sample = pd.DataFrame(index=pd.RangeIndex(44))
    entries = pd.DataFrame({"long_entry": False, "short_entry": False}, index=sample.index)
    entries.loc[0:2, "long_entry"] = True
    entries.loc[20, "short_entry"] = True
    result = GeneratedStrategy.generate_signals(None, sample, entries, None, p).data
    expected = np.zeros(44, dtype=int)
    expected[1:19] = 1
    expected[21:39] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
