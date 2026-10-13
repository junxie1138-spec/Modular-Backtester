from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletRecoveryParams:
    channel_window: int = 63
    energy_window: int = 5
    downside_share_max: float = 0.45
    max_position: float = 0.55
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletRecoveryParams]):
    strategy_id = "gen_local_1791311011"

    @classmethod
    def params_type(cls):
        return WaveletRecoveryParams

    def warmup_bars(self, params: WaveletRecoveryParams) -> int:
        return int(max(params.channel_window, 16 + params.energy_window) + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletRecoveryParams) -> pd.DataFrame:
        if params.channel_window < 2 or params.energy_window < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; channel_window must exceed one.")
        if not 0.0 < params.downside_share_max < 1.0 or not 0.15 < params.max_position < 1.0:
            raise ValueError("Energy and channel thresholds must lie within their valid ranges.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        energy_columns = []
        downside_columns = []
        for scale in (2, 4, 8):
            mean_column = f"mean_{scale}"
            haar_column = f"haar_{scale}"
            energy_column = f"energy_{scale}"
            downside_column = f"downside_energy_{scale}"
            out[mean_column] = out["log_close"].rolling(scale, min_periods=scale).mean()
            out[haar_column] = np.sqrt(scale / 2.0) * (out[mean_column] - out[mean_column].shift(scale))
            out[energy_column] = out[haar_column].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
            out[downside_column] = out[haar_column].clip(upper=0).pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
            energy_columns.append(energy_column)
            downside_columns.append(downside_column)
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=3)
        out["downside_energy"] = out[downside_columns].sum(axis=1, min_count=3)
        out["downside_share"] = out["downside_energy"] / out["total_energy"].where(out["total_energy"] > 0)
        out["prior_downside_share"] = out["downside_share"].shift(1)
        out["channel_low"] = data["low"].rolling(params.channel_window, min_periods=params.channel_window).min()
        out["channel_high"] = data["high"].rolling(params.channel_window, min_periods=params.channel_window).max()
        out["channel_range"] = out["channel_high"] - out["channel_low"]
        out["position"] = (data["close"] - out["channel_low"]) / out["channel_range"].where(out["channel_range"] > 0)
        out["prior_position"] = out["position"].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletRecoveryParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        wavelet_agrees = (
            (indicators["downside_share"] < params.downside_share_max)
            & (indicators["prior_downside_share"] >= params.downside_share_max)
        )
        position_agrees = (
            (indicators["position"] >= 0.15)
            & (indicators["position"] <= params.max_position)
            & (indicators["position"] > indicators["prior_position"])
        )
        entries = (wavelet_agrees & position_agrees).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    exit_bar = -1
                continue
            if entries[i]:
                signals[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        # Entry decision i fills at i+1; exit decision i+N fills at i+N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = WaveletRecoveryParams()
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"downside_share": 0.3, "prior_downside_share": 0.8, "position": 0.3, "prior_position": 0.2}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20] == 1
    assert result["size"].gt(0).all()
    indicators["position"] = 0.8
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()
