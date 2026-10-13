from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CUSUMParams:
    baseline_bars: int = 60
    drift_penalty: float = 0.2
    evidence_threshold: float = 3.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[CUSUMParams]):
    strategy_id = "gen_local-w2_1791281388"

    @classmethod
    def params_type(cls) -> type[CUSUMParams]:
        return CUSUMParams

    @staticmethod
    def warmup_bars(params: CUSUMParams) -> int:
        return max(2, int(params.baseline_bars)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: CUSUMParams) -> pd.DataFrame:
        if params.baseline_bars < 2 or params.holding_bars < 1:
            raise ValueError("Lookback must be at least two and holding duration positive.")
        if not np.isfinite(params.drift_penalty) or params.drift_penalty < 0:
            raise ValueError("Drift penalty must be finite and nonnegative.")
        if not np.isfinite(params.evidence_threshold) or params.evidence_threshold <= 0:
            raise ValueError("Evidence threshold must be finite and positive.")
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        baseline_mean = returns.rolling(params.baseline_bars, min_periods=params.baseline_bars).mean().shift(1)
        baseline_std = returns.rolling(params.baseline_bars, min_periods=params.baseline_bars).std(ddof=0).shift(1)
        innovation = ((returns - baseline_mean) / baseline_std.where(baseline_std > 0)).replace([np.inf, -np.inf], np.nan)
        increments = (innovation - params.drift_penalty).fillna(0.0)
        cumulative = increments.cumsum()
        evidence = (cumulative - cumulative.cummin().clip(upper=0.0)).where(innovation.notna())
        crossing = evidence.ge(params.evidence_threshold) & evidence.shift(1).lt(params.evidence_threshold)
        confirmation = crossing.shift(1, fill_value=False) & evidence.ge(params.evidence_threshold) & innovation.gt(params.drift_penalty)
        return pd.DataFrame({
            "returns": returns,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "innovation": innovation,
            "evidence": evidence,
            "crossing": crossing,
            "confirmation": confirmation.fillna(False),
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CUSUMParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("Holding duration must be positive.")
        entries = indicators["confirmation"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.holding_bars:
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif entries[i]:
                entry_bar = i
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = CUSUMParams(baseline_bars=10, evidence_threshold=2.0, holding_bars=5)
    returns = np.concatenate((np.tile([-0.001, 0.001], 30), np.full(3, 0.02), np.zeros(30)))
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=len(close), freq="B"))
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    signal = result["signal"].to_numpy()
    changes = np.diff(np.concatenate(([0], signal, [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    assert len(starts) > 0 and np.all(ends - starts == params.holding_bars)
    assert set(np.unique(signal)).issubset({0, 1}) and (result["size"] > 0).all()
    assert indicators["confirmation"].iloc[starts[0] - 1]
    assert indicators["crossing"].iloc[starts[0] - 2]
