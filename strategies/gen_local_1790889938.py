from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureAreaParams:
    lookback: int = 24
    strength_threshold: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureAreaParams]):
    strategy_id = "gen_local_1790889938"

    @classmethod
    def params_type(cls) -> type[SignatureAreaParams]:
        return SignatureAreaParams

    def warmup_bars(self, params: SignatureAreaParams) -> int:
        return max(2, int(params.lookback)) + 1

    def indicators(self, data: pd.DataFrame, params: SignatureAreaParams) -> pd.DataFrame:
        length = max(2, int(params.lookback))
        result = pd.DataFrame(index=data.index)
        result["log_close"] = np.log(data["close"].where(data["close"] > 0))
        result["log_return"] = result["log_close"].diff()
        result["path_variation"] = result["log_return"].abs().rolling(length, min_periods=length).sum()
        result["displacement"] = result["log_close"] - result["log_close"].shift(length)
        # Second-order antisymmetric signature of the time/log-price path.
        interior = result["log_close"].rolling(length - 1, min_periods=length - 1).sum().shift(1)
        result["interior_sum"] = interior
        area = 0.5 * (length - 1) * (result["log_close"] + result["log_close"].shift(length)) - interior
        result["signed_area"] = area
        denominator = result["path_variation"].where(result["path_variation"] > 0)
        result["path_area"] = (area / (length * denominator)).replace([np.inf, -np.inf], np.nan)
        result["efficiency"] = (result["displacement"] / denominator).clip(-1.0, 1.0)
        result["strength"] = (result["path_area"].clip(lower=0.0) * result["efficiency"].clip(lower=0.0)).fillna(0.0)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureAreaParams) -> SignalFrame:
        strength = indicators["strength"].fillna(0.0)
        eligible = ((strength > params.strength_threshold) & (indicators["displacement"] > 0)).fillna(False).to_numpy(dtype=bool)
        conviction_size = (0.25 + 0.70 * (strength / 0.15).clip(0.0, 1.0)).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        hold = max(1, int(params.hold_bars))
        exit_at = -1
        entry_size = 0.25
        for i in range(len(data)):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                    continue
                signal[i] = 1
                size[i] = entry_size
            elif eligible[i]:
                exit_at = i + hold
                entry_size = float(conviction_size[i])
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SignatureAreaParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=160, freq="B")
    close = 100.0 * np.exp(0.0002 * np.arange(len(index), dtype=float) ** 2)
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    indicators = strategy.indicators(data, params)
    output = strategy.generate_signals(data, indicators, None, params).data
    entries = np.flatnonzero(output["signal"].to_numpy() == 1)
    assert len(entries) > 0
    first = int(entries[0])
    assert first >= strategy.warmup_bars(params)
    assert output["signal"].iloc[first:first + params.hold_bars].eq(1).all()
    assert output["signal"].iloc[first + params.hold_bars] == 0
    assert output["signal"].isin([0, 1]).all()
    assert output["size"].gt(0).all()
    assert output["size"].iloc[first:first + params.hold_bars].nunique() == 1
    expected_size = 0.25 + 0.70 * np.clip(indicators["strength"].iloc[first - 1] / 0.15, 0.0, 1.0)
    assert np.isclose(output["size"].iloc[first], expected_size)
