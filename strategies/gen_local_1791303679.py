from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 12
    pressure_window: int = 4
    entropy_max: float = 0.65
    pressure_min: float = 0.15
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791303679"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.pattern_window < 2 or params.pressure_window < 1:
            raise ValueError("Lookbacks must be positive; pattern_window must be at least two.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        if not 0.0 <= params.entropy_max <= 1.0 or not 0.0 <= params.pressure_min <= 1.0:
            raise ValueError("Thresholds must lie in [0, 1].")
        return int(max(params.pattern_window + 2, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        a, b, c = data["close"].shift(2), data["close"].shift(1), data["close"]
        valid = (
            np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            & (a != b) & (a != c) & (b != c)
        )
        out["ordinal_code"] = (
            4 * (a < b).astype(int) + 2 * (b < c).astype(int) + (a < c).astype(int)
        ).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 2, 3, 4, 5, 7):
            probability = (
                out["ordinal_code"].eq(code).astype(float).where(valid)
                .rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            )
            out[f"motif_{code}"] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out["ordinal_entropy"] = entropy / np.log(6.0)
        out["body_volume"] = (data["close"] - data["open"]) * data["volume"]
        out["range_volume"] = (data["high"] - data["low"]) * data["volume"]
        valid_bar = (
            np.isfinite(data[["open", "high", "low", "close", "volume"]]).all(axis=1)
            & (data["volume"] > 0)
            & (data["high"] > data["low"])
            & (data["high"] >= data[["open", "close"]].max(axis=1))
            & (data["low"] <= data[["open", "close"]].min(axis=1))
        )
        out["body_flow"] = out["body_volume"].where(valid_bar).rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["range_flow"] = out["range_volume"].where(valid_bar).rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["intraday_pressure"] = (
            out["body_flow"] / out["range_flow"].where(out["range_flow"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        ordinal_agreement = (
            indicators["ordinal_entropy"].le(params.entropy_max)
            & indicators["motif_7"].ge(0.5)
            & indicators["motif_7"].gt(indicators["motif_0"])
        )
        pressure_agreement = indicators["intraday_pressure"].ge(params.pressure_min)
        eligible = (ordinal_agreement & pressure_agreement).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if i < exit_bar:
                signal[i] = 1
            elif i == exit_bar:
                # Force a flat bar; entry conditions cannot cancel a scheduled exit.
                signal[i] = 0
            elif eligible[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params(pattern_window=3, pressure_window=2)
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    close = np.arange(100.0, 112.0)
    data = pd.DataFrame(
        {"open": close - 0.5, "high": close + 0.5, "low": close - 1.0,
         "close": close, "volume": 1000.0},
        index=index,
    )
    indicators = strategy.indicators(data, params)
    assert indicators.index.equals(data.index)
    assert indicators["ordinal_entropy"].iloc[:4].isna().all()
    assert np.allclose(indicators["ordinal_entropy"].iloc[4:], 0.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
    blocked = indicators.copy()
    blocked["intraday_pressure"] = 0.0
    assert not strategy.generate_signals(data, blocked, None, params).data["signal"].any()
    blocked = indicators.copy()
    blocked["ordinal_entropy"] = 1.0
    assert not strategy.generate_signals(data, blocked, None, params).data["signal"].any()
    params.hold_bars = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
