from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.5
    entry_z: float = 1.75

    def __post_init__(self):
        if not np.isfinite(self.fractional_order) or not 0 < self.fractional_order < 1:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError("entry_z must be positive and finite")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791262454"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        # 64 filter observations + 63 previous filtered observations.
        return 127

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(np.isfinite(close) & (close > 0)))
        weights = np.ones(64, dtype=float)
        for k in range(1, 64):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: fixed 64-tap approximation; extend taps if tail sensitivity matters.
        filtered = np.full(len(data), np.nan)
        if len(data) >= 64:
            filtered[63:] = np.convolve(out["log_close"].to_numpy(), weights, mode="valid")
        out["fractional_close"] = filtered
        history = out["fractional_close"].shift(1).rolling(63, min_periods=63)
        out["fd_mean"] = history.mean()
        out["fd_std"] = history.std(ddof=0)
        out["fd_z"] = (out["fractional_close"] - out["fd_mean"]) / out["fd_std"].where(out["fd_std"] > 1e-12)
        out["absolute_step"] = out["log_close"].diff().abs()
        out["path_length"] = out["absolute_step"].rolling(20, min_periods=20).sum()
        out["net_displacement"] = out["log_close"].diff(20).abs()
        out["efficiency"] = out["net_displacement"] / out["path_length"].where(out["path_length"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        entries = ((indicators["fd_z"] <= -params.entry_z) &
                   (indicators["efficiency"] <= 0.30)).fillna(False).to_numpy()
        entries[:self.warmup_bars(params)] = False
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Reserve this bar for the mandatory exit, even if entry persists.
                    exit_bar = -1
            elif enter:
                signal[i] = 1
                exit_bar = i + 8
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=180, freq="B")
    close = 100 + np.arange(180) * 0.02 + np.sin(np.arange(180))
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000000.0}, index=index)
    strategy, params = GeneratedStrategy(), Params()
    ind = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(ind.iloc[:160], strategy.indicators(data.iloc[:160], params))
    assert ind["fd_z"].iloc[:126].isna().all()
    forced = ind.copy()
    forced["fd_z"], forced["efficiency"] = -3.0, 0.1
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=127)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[:128].eq(0).all()
    assert result["signal"].iloc[128:136].eq(1).all()
    assert result["signal"].iloc[136:137].eq(0).all()
    assert result["signal"].iloc[137:145].eq(1).all()
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0).all()
