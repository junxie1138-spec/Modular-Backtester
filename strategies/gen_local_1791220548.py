from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRelocationParams:
    recent_bars: int = 3
    baseline_bars: int = 4
    expansion_threshold: float = 1.5
    overlap_ceiling: float = 0.55
    migration_threshold: float = 0.5
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[RangeRelocationParams]):
    strategy_id = "gen_local_1791220548"

    @classmethod
    def params_type(cls) -> type[RangeRelocationParams]:
        return RangeRelocationParams

    def warmup_bars(self, params: RangeRelocationParams) -> int:
        if params.recent_bars < 1 or params.baseline_bars < 1:
            raise ValueError("Lookbacks must be positive")
        warmup = params.recent_bars + params.baseline_bars + 1
        if warmup > 10:
            raise ValueError("Warmup must not exceed ten bars")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("Holding period must be five to ten bars")
        if not np.isfinite(params.expansion_threshold) or params.expansion_threshold <= 1:
            raise ValueError("Expansion threshold must exceed one")
        if not np.isfinite(params.overlap_ceiling) or not 0 <= params.overlap_ceiling <= 1:
            raise ValueError("Overlap ceiling must be between zero and one")
        if not np.isfinite(params.migration_threshold) or params.migration_threshold <= 0:
            raise ValueError("Migration threshold must be positive")
        return int(warmup)

    def indicators(self, data: pd.DataFrame, params: RangeRelocationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["recent_range"] = out["range"].rolling(params.recent_bars).mean()
        out["baseline_range"] = out["range"].shift(params.recent_bars).rolling(params.baseline_bars).mean()
        out["expansion"] = out["recent_range"] / out["baseline_range"]
        out["previous_expansion"] = out["expansion"].shift(1)
        out["intersection"] = (
            pd.concat([data["high"], data["high"].shift(1)], axis=1).min(axis=1, skipna=False)
            - pd.concat([data["low"], data["low"].shift(1)], axis=1).max(axis=1, skipna=False)
        ).clip(lower=0)
        out["smaller_range"] = pd.concat(
            [out["range"], out["range"].shift(1)], axis=1
        ).min(axis=1, skipna=False)
        out["overlap"] = out["intersection"] / out["smaller_range"]
        out["mean_overlap"] = out["overlap"].rolling(params.recent_bars).mean()
        out["center"] = (data["high"] + data["low"]) / 2.0
        out["center_change"] = out["center"].diff()
        out["migration"] = out["center_change"].rolling(params.recent_bars).sum() / out["baseline_range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["migration"]).fillna(0).astype(int)
        out["event"] = (
            out["expansion"].gt(params.expansion_threshold)
            & out["previous_expansion"].le(params.expansion_threshold)
            & out["mean_overlap"].le(params.overlap_ceiling)
            & out["migration"].abs().ge(params.migration_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeRelocationParams,
    ) -> SignalFrame:
        events = indicators["event"].fillna(False).to_numpy(dtype=bool)
        directions = indicators["direction"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    continue
                signal[i] = position
            elif events[i] and directions[i] in (-1, 1):
                position = int(directions[i])
                exit_bar = i + params.hold_bars
                signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Opposing entry events cannot interrupt the seven-bar holding period.
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"event": True, "direction": [1] + [-1] * 9}, index=index
    )
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, RangeRelocationParams()
    )
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 0, -1]
    assert result.data["size"].gt(0).all()
