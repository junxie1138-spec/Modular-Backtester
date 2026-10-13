from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExcursionParams:
    ma_bars: int = 20
    regime_bars: int = 12
    entry_z: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ExcursionParams]):
    strategy_id = "gen_local_1791328788"

    @classmethod
    def params_type(cls):
        return ExcursionParams

    def warmup_bars(self, params: ExcursionParams) -> int:
        if params.ma_bars < 2 or params.regime_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if params.hold_bars not in (1, 2) or not np.isfinite(params.entry_z) or params.entry_z <= 0.75:
            raise ValueError("hold_bars must be one or two and entry_z must exceed 0.75")
        return int(params.ma_bars + params.regime_bars + 1)

    def indicators(self, data: pd.DataFrame, params: ExcursionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["excursion"] = out["z"].abs().ge(0.75).astype(float).where(out["z"].notna())
        out["previous_excursion"] = out["excursion"].shift(1)
        valid = out["excursion"].notna() & out["previous_excursion"].notna()
        out["susceptible"] = (1.0 - out["previous_excursion"]).where(valid)
        out["infected"] = out["previous_excursion"].where(valid)
        out["onset"] = ((out["previous_excursion"] == 0.0) & (out["excursion"] == 1.0)).astype(float).where(valid)
        out["recovery"] = ((out["previous_excursion"] == 1.0) & (out["excursion"] == 0.0)).astype(float).where(valid)
        for column in ("susceptible", "infected", "onset", "recovery"):
            out[column + "_count"] = out[column].rolling(params.regime_bars, min_periods=params.regime_bars).sum()
        out["onset_rate"] = out["onset_count"] / out["susceptible_count"].replace(0.0, np.nan)
        out["recovery_rate"] = out["recovery_count"] / out["infected_count"].replace(0.0, np.nan)
        out["rate_gap"] = out["onset_rate"] - out["recovery_rate"]
        out["conviction"] = ((out["z"].abs() / (2.0 * params.entry_z)).clip(0.0, 1.0) * (0.5 + 0.5 * out["rate_gap"].abs().clip(0.0, 1.0)))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExcursionParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        z = indicators["z"]
        gap = indicators["rate_gap"]
        eligible = np.isfinite(z) & np.isfinite(gap) & z.abs().ge(params.entry_z) & gap.ne(0.0)
        direction = np.where(gap > 0.0, np.sign(z), -np.sign(z))
        candidates = np.where(eligible, direction, 0).astype(int)
        proposed_size = (0.20 + 0.75 * indicators["conviction"].clip(0.0, 1.0)).fillna(0.20).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        position = 0
        entry_bar = -1
        entry_size = 0.20
        for i in range(len(data)):
            if position != 0 and i - entry_bar >= params.hold_bars:
                position = 0
                continue
            if position == 0 and candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
                entry_size = float(proposed_size[i])
            if position != 0:
                signals[i] = position
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"z": [2.0, -2.0] * 4, "rate_gap": 0.3, "conviction": [0.4, 0.8] * 4}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, ExcursionParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0.0).all()
    assert result["size"].iloc[1] == result["size"].iloc[2]
    assert result["size"].iloc[4] == result["size"].iloc[5]
    assert result["size"].iloc[4] > result["size"].iloc[1]
    derived["z"] = np.nan
    flat = GeneratedStrategy.generate_signals(None, data, derived, None, ExcursionParams()).data
    assert flat["signal"].eq(0).all()
