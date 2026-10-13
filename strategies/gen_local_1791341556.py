from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransportParams:
    range_bars: int = 8
    wave_bars: int = 4
    reversal_threshold: float = 0.06


class GeneratedStrategy(BaseStrategy[GapTransportParams]):
    strategy_id = "gen_local_1791341556"

    @classmethod
    def params_type(cls) -> type[GapTransportParams]:
        return GapTransportParams

    def warmup_bars(self, params: GapTransportParams) -> int:
        if params.range_bars < 2 or params.wave_bars < 1:
            raise ValueError("Lookbacks must be positive; range_bars must be at least two.")
        if not np.isfinite(params.reversal_threshold) or params.reversal_threshold <= 0:
            raise ValueError("reversal_threshold must be finite and positive.")
        return int(max(200, params.range_bars + params.wave_bars))

    def indicators(self, data: pd.DataFrame, params: GapTransportParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["prior_high"] = data["high"].shift(1).rolling(params.range_bars).max()
        out["prior_low"] = data["low"].shift(1).rolling(params.range_bars).min()
        out["range_width"] = (out["prior_high"] - out["prior_low"]).replace(0.0, np.nan)
        out["gap"] = data["open"] - out["previous_close"]
        out["position"] = ((out["previous_close"] - out["prior_low"]) / out["range_width"]).clip(0.0, 1.0)
        out["available_space"] = out["position"].where(out["gap"] < 0.0, 1.0 - out["position"])
        out["transport"] = out["gap"] / out["range_width"] * out["available_space"]
        out["wave"] = out["transport"].rolling(params.wave_bars).sum()
        out["ma200"] = data["close"].rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTransportParams,
    ) -> SignalFrame:
        wave = indicators["wave"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        valid = np.isfinite(wave) & np.isfinite(close) & np.isfinite(regime)
        direction = np.zeros(len(data), dtype=int)
        direction[valid & (wave > params.reversal_threshold)] = 1
        direction[valid & (wave < -params.reversal_threshold)] = -1
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if valid[i]:
                if position != 0:
                    # Opposite gap evidence exits to cash; neutral evidence holds.
                    # The MA gates entries only, so it cannot force another exit.
                    if direction[i] == -position:
                        position = 0
                elif direction[i] == 1 and close[i] > regime[i]:
                    position = 1
                elif direction[i] == -1 and close[i] < regime[i]:
                    position = -1
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapTransportParams()
    strategy = GeneratedStrategy()
    data = pd.DataFrame(
        {"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=207, freq="B"),
    )
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived["wave"].first_valid_index() == data.index[params.range_bars + params.wave_bars - 1]
    derived.loc[data.index[-7:], "wave"] = [0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1]
    derived.loc[data.index[-7:], "ma200"] = [100.0, 100.0, 100.0, 102.0, 102.0, 102.0, 100.0]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[-7:].tolist() == [0, 1, 1, 1, 0, -1, -1]
    assert result["signal"].iloc[:200].eq(0).all()
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
