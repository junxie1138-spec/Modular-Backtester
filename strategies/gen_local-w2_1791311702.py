from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    slope_window: int = 12
    noise_window: int = 20
    innovation_threshold: float = 0.65
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local-w2_1791311702"

    @classmethod
    def params_type(cls) -> type[InnovationParams]:
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        if params.slope_window < 3 or params.noise_window < 2:
            raise ValueError("Indicator windows are too short")
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between 3 and 5")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be positive and finite")
        if not np.isfinite(params.profit_target) or not 0 < params.profit_target < 1:
            raise ValueError("profit_target must be between zero and one")
        return int(max(200, params.slope_window + 1, params.noise_window + 2))

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        log_close = np.log(close)
        n = params.slope_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2
        slope = pd.Series(np.nan, index=data.index, dtype=float)
        if len(data) >= n:
            slope.iloc[n - 1:] = np.convolve(
                log_close.to_numpy(), weights[::-1], mode="valid"
            ) / np.dot(weights, weights)
        level = log_close.rolling(n, min_periods=n).mean() + slope * (n - 1) / 2
        forecast = level.shift(1) + slope.shift(1)
        innovation = log_close - forecast
        noise = log_close.diff().rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).shift(1)
        score = (innovation / noise.where(noise > 1e-8)).replace(
            [np.inf, -np.inf], np.nan
        )
        ma200 = close.rolling(200, min_periods=200).mean()
        threshold = params.innovation_threshold
        long_entry = (
            (close > ma200) & (slope.shift(1) > 0)
            & (score > threshold) & (score.shift(1) <= threshold)
        )
        short_entry = (
            (close < ma200) & (slope.shift(1) < 0)
            & (score < -threshold) & (score.shift(1) >= -threshold)
        )
        return pd.DataFrame({
            "log_close": log_close,
            "slope": slope,
            "level": level,
            "forecast": forecast,
            "innovation": innovation,
            "noise": noise,
            "score": score,
            "ma200": ma200,
            "entry": np.select([long_entry, short_entry], [1, -1], default=0).astype(int),
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: InnovationParams
    ) -> SignalFrame:
        self.warmup_bars(params)
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # A decision on the preceding close fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                )
                gain = position * (closes[i] / entry_price - 1) if valid else 0.0
                age = i - entry_bar + 1
                if not valid or gain >= params.profit_target or age >= params.max_hold:
                    position = 0
                    entry_price = np.nan
                    # Preserve a flat decision on exit; no same-close reentry.
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = InnovationParams(profit_target=0.01)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=14, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 97.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.iloc[2, data.columns.get_loc("close")] = 102.0
    data.iloc[6, data.columns.get_loc("close")] = 98.0
    entries = pd.DataFrame({"entry": [1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=260, freq="D")
    t = np.arange(260)
    close = 100 * np.exp(0.0005 * t + 0.01 * np.sin(t / 3))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:240], params)
    pd.testing.assert_frame_equal(full.iloc[:240], prefix)
    assert (full["entry"].iloc[:199] == 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
