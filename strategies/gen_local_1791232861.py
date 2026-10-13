from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    min_streak: int = 2
    shape_tolerance: float = 0.5
    forecast_threshold: float = 0.2
    profit_target: float = 0.006
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791232861"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        # Three return signs, shifted six bars, require ten closes.
        return 10

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if params.min_streak not in (2, 3) or params.max_hold not in (1, 2):
            raise ValueError("min_streak must be 2 or 3; max_hold must be 1 or 2")
        values = [params.shape_tolerance, params.forecast_threshold, params.profit_target]
        if not np.isfinite(values).all() or params.shape_tolerance < 0 or min(values[1:]) <= 0:
            raise ValueError("Invalid analog thresholds or profit target")
        out = pd.DataFrame(index=data.index)
        out["return"] = (data["close"] / data["close"].shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["return"]).fillna(0)
        same_two = out["direction"].ne(0) & out["direction"].eq(out["direction"].shift(1))
        same_three = same_two & out["direction"].eq(out["direction"].shift(2))
        # Saturate at three: this is a finite three-return shape, not an unbounded lookback.
        out["streak"] = out["direction"] * (1 + same_two.astype(int) + same_three.astype(int))
        out["shape"] = (out["return"].abs() / out["return"].shift(1).abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        for lag in (4, 5, 6):
            out[f"streak_{lag}"] = out["streak"].shift(lag)
            out[f"shape_{lag}"] = out["shape"].shift(lag)
            out[f"outcome_{lag}"] = out["return"].shift(lag - 1)
            out[f"distance_{lag}"] = (out["shape"] - out[f"shape_{lag}"]).abs()
            match = (
                out["streak"].eq(out[f"streak_{lag}"])
                & out["streak"].abs().ge(params.min_streak)
                & out[f"distance_{lag}"].le(params.shape_tolerance)
            )
            out[f"scaled_outcome_{lag}"] = (
                out[f"outcome_{lag}"] / out["return"].shift(lag).abs().replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).clip(-3, 3)
            out[f"weight_{lag}"] = (
                1 - out[f"distance_{lag}"] / (params.shape_tolerance + 1e-12)
            ).clip(0, 1).where(match & out[f"scaled_outcome_{lag}"].notna(), 0).fillna(0)
            numerator += out[f"weight_{lag}"] * out[f"scaled_outcome_{lag}"].fillna(0)
            denominator += out[f"weight_{lag}"]
        out["analog_weight"] = denominator
        out["forecast"] = (numerator / denominator.replace(0, np.nan)).fillna(0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"].fillna(0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if position:
                # Entry decided yesterday is filled at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1) if valid else 0.0
                # Targets are checked at the close; flatten at the next open.
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                signals[i] = position
                continue
            if np.isfinite(closes[i]) and closes[i] > 0 and abs(forecast[i]) >= params.forecast_threshold:
                position = 1 if forecast[i] > 0 else -1
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=22)
    data = pd.DataFrame({c: np.full(22, 100.0) for c in ("open", "high", "low", "close", "volume")}, index=index)
    data.loc[index[10], "close"] = 101.0
    data.loc[index[14], "close"] = 99.0
    strategy, params = GeneratedStrategy(), AnalogParams()
    actual = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(actual.iloc[:16], strategy.indicators(data.iloc[:16], params))
    forced = pd.DataFrame({"forecast": np.zeros(22)}, index=index)
    forced.loc[index[[9, 13, 17]], "forecast"] = [1.0, -1.0, 1.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[[10, 11, 14, 15, 18, 19, 20]].tolist() == [1, 0, -1, 0, 1, 1, 0]
    assert result["signal"].iloc[:10].eq(0).all()
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
