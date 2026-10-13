from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapWaveletParams:
    fine_energy_share: float = 0.72
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapWaveletParams]):
    strategy_id = "gen_local_1790996505"

    @classmethod
    def params_type(cls) -> type[GapWaveletParams]:
        return GapWaveletParams

    def warmup_bars(self, params: GapWaveletParams) -> int:
        return 33

    def indicators(self, data: pd.DataFrame, params: GapWaveletParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        previous_close = data["close"].where(data["close"] > 0).shift(1)
        opening = data["open"].where(data["open"] > 0)
        gap = np.log(opening / previous_close).replace([np.inf, -np.inf], np.nan)
        result["gap"] = gap
        result["gap_rms"] = np.sqrt(gap.pow(2).rolling(32, min_periods=32).mean())
        result["recovered"] = (data["close"] >= previous_close) & previous_close.notna()
        for scale in (2, 4, 8, 16):
            half = scale // 2
            recent = gap.rolling(half, min_periods=half).mean()
            detail = np.sqrt(scale) * (recent - recent.shift(half)) / 2.0
            result[f"haar_{scale}"] = detail
            result[f"energy_{scale}"] = detail.pow(2).rolling(16, min_periods=16).mean()
        fine = result["energy_2"] + result["energy_4"]
        total = fine + result["energy_8"] + result["energy_16"]
        result["fine_share"] = (fine / total.where(total > 0)).replace([np.inf, -np.inf], np.nan)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapWaveletParams,
    ) -> SignalFrame:
        eligible = (
            (indicators["fine_share"] >= params.fine_energy_share)
            & (indicators["gap_rms"] > 0)
            & (indicators["gap"] < -indicators["gap_rms"])
            & indicators["recovered"].fillna(False)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        holding_bars = max(1, int(params.holding_bars))
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif eligible[i]:
                signal[i] = 1
                exit_bar = i + holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapWaveletParams()
    index = pd.date_range("2020-01-01", periods=50, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived["fine_share"].isna().all()
    synthetic = pd.DataFrame({"gap": -0.02, "gap_rms": 0.01, "fine_share": 0.9, "recovered": True}, index=index)
    output = strategy.generate_signals(data, synthetic, None, params).data
    assert output["signal"].iloc[:21].tolist() == [0] + [1] * 18 + [0, 1]
    assert output["size"].gt(0).all()
