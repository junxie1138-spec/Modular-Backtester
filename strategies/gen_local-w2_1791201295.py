from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    tail_percentile: float = 0.20
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local-w2_1791201295"

    @classmethod
    def params_type(cls):
        return RotationParams

    def warmup_bars(self, params: RotationParams) -> int:
        if not (0.0 < params.tail_percentile < 0.5):
            raise ValueError("tail_percentile must be between zero and one-half")
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0.0):
            raise ValueError("trail_atr must be finite and positive")
        # Previous close + ten returns + 252 divergence observations.
        return 262

    def indicators(self, data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        opening = data["open"].where(data["open"] > 0.0)
        previous = close.shift(1)
        out["overnight"] = np.log(opening / previous)
        out["intraday"] = np.log(close / opening)
        out["divergence"] = (out["overnight"] - out["intraday"]).rolling(10, min_periods=10).sum()
        out["percentile"] = out["divergence"].rolling(252, min_periods=252).rank(pct=True)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        month = pd.Series(data.index.to_period("M"), index=data.index)
        out["month_turn"] = month.ne(month.shift(1)) & month.shift(1).notna()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RotationParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        features = indicators[["percentile", "atr"]].to_numpy(dtype=float)
        turns = indicators["month_turn"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry, best, floor = np.nan, 0.0, -np.inf
        armed = False

        for i in range(len(data)):
            opening, high, low, close = prices[i]
            percentile, atr = features[i]
            valid_price = np.isfinite(prices[i]).all() and min(opening, high, low, close) > 0.0
            if position:
                # The previous close requested entry; observe its fill only now.
                if np.isnan(entry):
                    entry, entry_bar = opening, i
                if not valid_price or not np.isfinite(entry) or entry <= 0.0:
                    position, entry = 0, np.nan
                else:
                    adverse = low if position == 1 else high
                    adverse_return = position * (adverse / entry - 1.0)
                    # Test the previously established floor before updating it.
                    stopped = armed and adverse_return <= floor
                    if stopped or i - entry_bar + 1 >= 18:
                        position, entry = 0, np.nan
                    else:
                        gain = position * (close / entry - 1.0)
                        best = max(best, gain)
                        armed = armed or best >= 0.02
                        if armed and np.isfinite(atr) and atr >= 0.0:
                            # Signed-return floor always rises. A short's price cap falls.
                            floor = max(floor, 0.0, best - params.trail_atr * atr / entry)
                # An exit cannot re-enter on the same decision bar.
            elif i >= warmup - 1 and turns[i] and valid_price and np.isfinite(features[i]).all() and atr > 0.0:
                if percentile <= params.tail_percentile:
                    position = 1
                elif percentile >= 1.0 - params.tail_percentile:
                    position = -1
                if position:
                    entry, best, floor, armed = np.nan, 0.0, -np.inf, False
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = RotationParams()
    index = pd.bdate_range("2020-01-01", periods=300)
    for direction in (1, -1):
        for activate in (False, True):
            close = np.full(300, 100.0)
            if activate:
                close[264] = 100.0 * (1.0 + direction * 0.03)
            data = pd.DataFrame({
                "open": 100.0, "high": close + 0.1,
                "low": close - 0.1, "close": close, "volume": 1000.0,
            }, index=index)
            data["high"] = data[["high", "open"]].max(axis=1)
            data["low"] = data[["low", "open"]].min(axis=1)
            features = pd.DataFrame({
                "percentile": 0.1 if direction == 1 else 0.9,
                "atr": 1.0, "month_turn": False,
            }, index=index)
            features.iloc[261, features.columns.get_loc("month_turn")] = True
            result = strategy.generate_signals(data, features, None, params).data
            assert result["signal"].iloc[261] == 0
            assert result["signal"].iloc[262] == direction
            exit_bar = 266 if activate else 280
            assert result["signal"].iloc[exit_bar - 1] == direction
            assert result["signal"].iloc[exit_bar] == 0
            prefix = strategy.generate_signals(data.iloc[:265], features.iloc[:265], None, params).data
            assert prefix.equals(result.iloc[:265])
            assert result["size"].gt(0.0).all()
    calculated = strategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert calculated["percentile"].iloc[:261].isna().all()
    assert np.isfinite(calculated["percentile"].iloc[261])
    assert strategy.warmup_bars(params) == 262


if __name__ == "__main__":
    _self_check()
