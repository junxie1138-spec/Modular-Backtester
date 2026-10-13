from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletCompressionParams:
    reference_window: int = 40
    compression_ratio: float = 0.55
    release_multiplier: float = 1.8
    atr_window: int = 14
    trailing_atr: float = 1.25
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletCompressionParams]):
    strategy_id = "gen_local-w2_1791254279"

    @classmethod
    def params_type(cls) -> type[WaveletCompressionParams]:
        return WaveletCompressionParams

    def warmup_bars(self, params: WaveletCompressionParams) -> int:
        return int(max(200, params.reference_window + 11, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletCompressionParams) -> pd.DataFrame:
        if not (
            params.reference_window >= 2
            and params.atr_window >= 1
            and 1 <= params.holding_bars <= 2
            and np.isfinite([params.compression_ratio, params.release_multiplier, params.trailing_atr]).all()
            and 0 < params.compression_ratio < 1
            and params.release_multiplier > 1
            and params.trailing_atr > 0
        ):
            raise ValueError("Invalid wavelet compression parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        for half_width in (1, 2, 4):
            label = str(2 * half_width)
            out["smooth_" + label] = out["log_close"].rolling(half_width, min_periods=half_width).mean()
            out["detail_" + label] = np.sqrt(half_width / 2.0) * (
                out["smooth_" + label] - out["smooth_" + label].shift(half_width)
            )
            out["energy_" + label] = out["detail_" + label].pow(2).rolling(3, min_periods=3).mean()
            out["reference_" + label] = out["energy_" + label].shift(1).rolling(
                params.reference_window, min_periods=params.reference_window
            ).mean()
            out["ratio_" + label] = out["energy_" + label] / out["reference_" + label].where(
                out["reference_" + label] > 0
            )
        out["compressed"] = out[["ratio_2", "ratio_4", "ratio_8"]].le(params.compression_ratio).all(axis=1)
        out["entry"] = (
            out["compressed"].shift(1, fill_value=False)
            & out["ratio_2"].gt(params.compression_ratio)
            & out["energy_2"].ge(params.release_multiplier * out["energy_2"].shift(1))
            & out["ratio_4"].le(params.compression_ratio)
            & out["ratio_8"].le(params.compression_ratio)
            & out["detail_2"].gt(0)
            & close.gt(out["ma200"])
            & out["atr"].gt(0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletCompressionParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                held += 1
                if valid:
                    high_water = close if held == 1 else max(high_water, close)
                    stop = max(stop, high_water - params.trailing_atr * atr)
                if not valid or close <= stop or held >= params.holding_bars:
                    active = False
                    held = 0
                    high_water = np.nan
                    stop = -np.inf
                else:
                    signals[i] = 1
            elif valid and entries[i]:
                active = True
                held = 0
                high_water = np.nan
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletCompressionParams()
    index = pd.date_range("2020-01-01", periods=6)
    close = np.array([100.0, 104.0, 102.0, 101.0, 101.0, 101.0])
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close,
                         "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0,
                            "entry": [True, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert result.index.equals(data.index) and result["size"].gt(0).all()
    params.holding_bars = 1
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()
    assert strategy.warmup_bars(params) >= 200
