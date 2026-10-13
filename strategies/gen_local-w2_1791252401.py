from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 12
    atr_mult: float = 0.8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791252401"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return max(params.lookback + 2, 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.lookback < 3 or not np.isfinite(params.atr_mult) or params.atr_mult <= 0:
            raise ValueError("lookback must be >= 3 and atr_mult must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        for lag in (1, 2, 3):
            out[f"lag{lag}"] = out["return"].shift(lag)
        a, b, c, d = (out["lag3"], out["lag2"], out["lag1"], out["return"])
        # Second-level antisymmetric signature of three (lagged return, return) increments.
        out["signature_area"] = 0.5 * (a * c - b * b + a * d - b * c + b * d - c * c)
        out["autocorrelation"] = out["return"].rolling(
            params.lookback, min_periods=params.lookback
        ).corr(out["lag1"])
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.lookback, min_periods=params.lookback
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams
    ) -> SignalFrame:
        ready = indicators[["signature_area", "autocorrelation", "return", "lag1", "atr"]].notna().all(axis=1)
        entry = (
            ready & (indicators["signature_area"] < 0)
            & (indicators["autocorrelation"] < 0)
            & (indicators["return"] > 0) & (indicators["lag1"] < 0)
            & (indicators["atr"] > 0)
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = stop = -np.inf
        for i in range(len(data)):
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    active = False
                else:
                    # Only closes observed after the next-open entry count toward the watermark.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.atr_mult * atrs[i])
                    if closes[i] <= stop or age >= 2:
                        active = False
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                age = 0
                high_water = stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=64)
    close = 100.0 * np.exp(np.cumsum(0.01 * np.sin(np.arange(64) * 1.7)))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000.0}, index=index
    )
    strategy = GeneratedStrategy()
    ind = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(ind.iloc[:40], strategy.indicators(data.iloc[:40], params))
    result = strategy.generate_signals(data, ind, None, params).data
    prefix = strategy.generate_signals(data.iloc[:40], ind.iloc[:40], None, params).data
    pd.testing.assert_frame_equal(result.iloc[:40], prefix)
    assert result["signal"].isin([0, 1]).all() and (result["size"] > 0).all()
    assert not result["signal"].iloc[:strategy.warmup_bars(params)].any()
    fake = pd.DataFrame(
        {"signature_area": [-1.0] + [0.0] * 63, "autocorrelation": -0.5,
         "return": 0.01, "lag1": -0.01, "atr": 1.0}, index=index
    )
    held = strategy.generate_signals(data, fake, None, params).data["signal"]
    assert held.iloc[:5].tolist() == [0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
