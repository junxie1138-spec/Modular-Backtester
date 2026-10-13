from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalSeasonalityParams:
    pattern_window: int = 6
    entropy_mid: float = 0.65
    entropy_band: float = 0.05


class GeneratedStrategy(BaseStrategy[OrdinalSeasonalityParams]):
    strategy_id = "gen_local_1791222651"

    @classmethod
    def params_type(cls):
        return OrdinalSeasonalityParams

    def warmup_bars(self, params: OrdinalSeasonalityParams) -> int:
        if not 3 <= params.pattern_window <= 7:
            raise ValueError("pattern_window must be between 3 and 7")
        if not 0.0 <= params.entropy_band < params.entropy_mid < 1.0 - params.entropy_band:
            raise ValueError("entropy thresholds must lie strictly inside (0, 1)")
        return int(params.pattern_window + 3)

    def indicators(self, data: pd.DataFrame, params: OrdinalSeasonalityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        valid = a.notna() & b.notna() & c.notna()
        out["pattern"] = (4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)).where(valid)
        # Equal returns are ordered by their original chronological position.
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = out["pattern"].eq(code).astype(float).where(valid).rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            out[f"probability_{code}"] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=np.finfo(float).tiny))
        out["entropy"] = entropy / np.log(6.0)
        out["calendar_side"] = np.where(data.index.dayofweek % 2 == 0, 1, -1)
        out["entry"] = 0
        ordered = out["entropy"].le(params.entropy_mid - params.entropy_band)
        disordered = out["entropy"].ge(params.entropy_mid + params.entropy_band)
        out.loc[ordered, "entry"] = out.loc[ordered, "calendar_side"]
        out.loc[disordered, "entry"] = -out.loc[disordered, "calendar_side"]
        out["entry"] = out["entry"].astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalSeasonalityParams) -> SignalFrame:
        state = 0
        positions = np.zeros(len(data), dtype=int)
        for i, desired in enumerate(indicators["entry"].fillna(0).astype(int).to_numpy()):
            if state == 0:
                state = int(desired)
            elif desired == -state:
                state = 0
            positions[i] = state
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = OrdinalSeasonalityParams()
    index = pd.bdate_range("2024-01-01", periods=16)
    close = np.exp(np.arange(16, dtype=float) ** 2 / 1000.0) * 100.0
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 9
    assert derived["entropy"].iloc[:8].isna().all()
    assert np.allclose(derived["entropy"].iloc[8:], 0.0)
    entries = pd.DataFrame({"entry": [1, 1, -1, -1, 0, 1, 1]}, index=index[:7])
    result = strategy.generate_signals(data.iloc[:7], entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
