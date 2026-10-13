from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    dominance: float = 0.60
    min_move: float = 0.001
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790937137"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not (0.5 <= params.dominance <= 1.0):
            raise ValueError("dominance must be between 0.5 and 1")
        if not np.isfinite(params.min_move) or params.min_move <= 0:
            raise ValueError("min_move must be finite and positive")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        out = pd.DataFrame(index=data.index)
        close = pd.to_numeric(data["close"], errors="coerce")
        out["return"] = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        out["pair_sum"] = r.rolling(2, min_periods=2).sum()
        out["quad_sum"] = r.rolling(4, min_periods=4).sum()
        out["oct_sum"] = r.rolling(8, min_periods=8).sum()
        out["detail_2"] = (r - r.shift(1)) / np.sqrt(2.0)
        out["detail_4"] = (out["pair_sum"] - out["pair_sum"].shift(2)) / 2.0
        out["detail_8"] = (out["quad_sum"] - out["quad_sum"].shift(4)) / np.sqrt(8.0)
        out["approx_8"] = out["oct_sum"] / np.sqrt(8.0)
        fine = out["detail_2"].pow(2)
        medium = out["detail_4"].pow(2)
        out["fine_energy"] = fine + fine.shift(2) + fine.shift(4) + fine.shift(6)
        out["medium_energy"] = medium + medium.shift(4)
        out["coarse_energy"] = out["detail_8"].pow(2)
        out["trend_energy"] = out["approx_8"].pow(2)
        out["total_energy"] = out[["fine_energy", "medium_energy", "coarse_energy", "trend_energy"]].sum(axis=1, min_count=4)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        out["trend_share"] = (out["trend_energy"] / denominator).fillna(0.0)
        out["fine_share"] = (out["fine_energy"] / denominator).fillna(0.0)
        out["trend_move"] = (out["oct_sum"] / 8.0).fillna(0.0)
        out["ready"] = denominator.notna()
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        trend = indicators["ready"] & (indicators["trend_share"] >= params.dominance)
        reversal = indicators["ready"] & (indicators["fine_share"] >= params.dominance)
        candidates = np.zeros(len(data), dtype=int)
        trend_ok = trend & (indicators["trend_move"].abs() >= params.min_move)
        reversal_ok = reversal & (indicators["return"].abs() >= params.min_move)
        candidates[trend_ok.to_numpy()] = np.sign(indicators.loc[trend_ok, "trend_move"]).astype(int)
        candidates[reversal_ok.to_numpy()] = -np.sign(indicators.loc[reversal_ok, "return"]).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(candidates):
            if position and i >= exit_at:
                position = 0
                continue
            if position == 0 and candidate != 0:
                position = int(candidate)
                exit_at = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletParams()
    idx = pd.date_range("2020-01-01", periods=16)
    close = pd.Series(np.exp(np.arange(16) * 0.003), index=idx)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0})
    ind = strategy.indicators(data, params)
    assert not ind["ready"].iloc[:8].any()
    assert np.allclose(ind["total_energy"].iloc[8:], ind["return"].pow(2).rolling(8).sum().iloc[8:])
    ctx = StrategyContext("SPY", "1d", 9)
    ind["trend_move"] = 0.003
    ind.loc[idx[9:14], "trend_move"] = -0.003
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0] * 9 + [1, 1, 0, -1, -1, 0, 1]
    assert (result["size"] > 0).all()
