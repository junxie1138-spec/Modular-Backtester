from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_bars: int = 6
    area_threshold: float = 0.03
    atr_bars: int = 14
    trail_atr: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791207902"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(220, params.signature_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.signature_bars < 2 or params.atr_bars < 1:
            raise ValueError("Indicator windows must be positive; signature_bars must be at least two")
        if not np.isfinite(params.area_threshold) or params.area_threshold < 0:
            raise ValueError("area_threshold must be finite and nonnegative")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        n = params.signature_bars
        out["price_step"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volume_step"] = np.log(data["volume"].where(data["volume"] > 0)).diff().replace([np.inf, -np.inf], np.nan)
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, n):
            name = f"area_pairs_{lag}"
            out[name] = (out["price_step"].shift(lag) * out["volume_step"] - out["volume_step"].shift(lag) * out["price_step"]).rolling(n - lag, min_periods=n - lag).sum()
            area = area + out[name]
        out["price_variation"] = out["price_step"].abs().rolling(n, min_periods=n).sum()
        out["volume_variation"] = out["volume_step"].abs().rolling(n, min_periods=n).sum()
        denominator = out["price_variation"] * out["volume_variation"]
        out["signature_area"] = (0.5 * area / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        out["displacement"] = out["price_step"].rolling(n, min_periods=n).sum()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["ma200_slope"] = out["ma200"].diff(20)
        out["ready"] = np.isfinite(out[["signature_area", "displacement", "atr", "ma200", "ma200_slope"]]).all(axis=1) & np.isfinite(close) & (out["atr"] > 0)
        out["eligible"] = out["ready"] & (close > out["ma200"])
        out["rising_regime"] = out["ma200_slope"] > 0
        continuation = out["rising_regime"] & (out["signature_area"] < -params.area_threshold) & (out["displacement"] > 0)
        exhaustion = ~out["rising_regime"] & (out["signature_area"] > params.area_threshold) & (out["displacement"] < 0)
        out["entry"] = (out["eligible"] & (continuation | exhaustion)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            if active:
                age += 1
                if not eligible[i] or not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    active = False
                else:
                    # Entry fills at this bar's open; only subsequent closes enter the high-water mark.
                    high_water = closes[i] if age == 1 else max(high_water, closes[i])
                    candidate = high_water - params.trail_atr * atrs[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    if closes[i] <= stop or age >= params.holding_bars:
                        active = False
            elif entries[i] and eligible[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = np.nan
            decisions[i] = int(active)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    sample = pd.DataFrame({"close": [110.0, 112.0, 111.0, 108.0, 108.0]}, index=index)
    features = pd.DataFrame({"atr": 1.0, "eligible": True, "entry": [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, features, None, SignatureParams(trail_atr=2.0))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
    features["atr"] = [1.0, 1.0, 10.0, 10.0, 10.0]
    sample["close"] = [110.0, 112.0, 109.0, 109.0, 109.0]
    result = strategy.generate_signals(sample, features, None, SignatureParams(trail_atr=2.0))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    sample["close"] = [110.0, 112.0, 113.0, 114.0, 115.0]
    result = strategy.generate_signals(sample, features, None, SignatureParams(holding_bars=3))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
