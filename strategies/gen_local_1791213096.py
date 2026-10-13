from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletBreadthParams:
    min_effective_scales: float = 2.2
    min_drawdown: float = 0.012
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletBreadthParams]):
    strategy_id = "gen_local_1791213096"

    @classmethod
    def params_type(cls) -> type[WaveletBreadthParams]:
        return WaveletBreadthParams

    def warmup_bars(self, params: WaveletBreadthParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveletBreadthParams) -> pd.DataFrame:
        log_close = np.log(data["close"].where(data["close"] > 0))
        returns = log_close.diff()
        r = [returns.shift(k) for k in range(8)]
        fine = [(r[k] - r[k + 1]) / np.sqrt(2.0) for k in (0, 2, 4, 6)]
        middle = [(r[k] + r[k + 1] - r[k + 2] - r[k + 3]) / 2.0 for k in (0, 4)]
        coarse = (r[0] + r[1] + r[2] + r[3] - r[4] - r[5] - r[6] - r[7]) / np.sqrt(8.0)
        fine_energy = sum(x.pow(2) for x in fine)
        middle_energy = sum(x.pow(2) for x in middle)
        coarse_energy = coarse.pow(2)
        total_energy = fine_energy + middle_energy + coarse_energy
        denominator = fine_energy.pow(2) + middle_energy.pow(2) + coarse_energy.pow(2)
        effective_scales = total_energy.pow(2).div(denominator.where(denominator > 0))
        cumulative_return = returns.rolling(8, min_periods=8).sum()
        ready = cumulative_return.notna() & effective_scales.notna()
        entry_ready = (
            ready
            & (effective_scales >= params.min_effective_scales)
            & (cumulative_return <= -params.min_drawdown)
            & (returns > 0)
        )
        return pd.DataFrame({
            "log_return": returns,
            "fine_energy": fine_energy,
            "middle_energy": middle_energy,
            "coarse_energy": coarse_energy,
            "total_energy": total_energy,
            "effective_scales": effective_scales,
            "cumulative_return": cumulative_return,
            "ready": ready,
            "entry_ready": entry_ready,
        }, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletBreadthParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar >= params.hold_bars:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entries[i]:
                entry_bar = i
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletBreadthParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    prices = np.arange(100.0, 124.0)
    data = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived.isna().any().any()
    assert not derived["entry_ready"].iloc[:8].any()
    forced_entries = pd.DataFrame({"entry_ready": True}, index=index)
    result = strategy.generate_signals(data, forced_entries, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20] == 1
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()
