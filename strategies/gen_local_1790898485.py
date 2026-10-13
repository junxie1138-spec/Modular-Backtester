from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEchoParams:
    estimate_bars: int = 60
    impulse_bars: int = 5
    atr_bars: int = 14
    min_score: float = 0.04
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapEchoParams]):
    strategy_id = "gen_local_1790898485"

    @classmethod
    def params_type(cls):
        return GapEchoParams

    def warmup_bars(self, params: GapEchoParams) -> int:
        return int(max(params.estimate_bars + 2, params.impulse_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapEchoParams) -> pd.DataFrame:
        if min(params.estimate_bars, params.impulse_bars, params.atr_bars, params.hold_bars) < 2:
            raise ValueError("Lookbacks and holding period must be at least two.")
        if not np.isfinite(params.min_score) or params.min_score <= 0:
            raise ValueError("min_score must be finite and positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        prev = data["close"].shift(1).replace(0, np.nan)
        returns = data["close"].div(prev).sub(1)
        gap = data["open"].div(prev).sub(1)
        lag_gap = gap.shift(1)
        window = params.estimate_bars
        variance = lag_gap.rolling(window).var().replace(0, np.nan)
        covariance = lag_gap.rolling(window).cov(returns)
        beta = covariance.div(variance)
        impulse = gap.rolling(params.impulse_bars).mean()
        noise = returns.rolling(window).std().replace(0, np.nan)
        score = beta.mul(impulse).div(noise)
        tr = pd.concat([
            data["high"].sub(data["low"]),
            data["high"].sub(prev).abs(),
            data["low"].sub(prev).abs(),
        ], axis=1).max(axis=1)
        return pd.DataFrame({
            "previous_close": prev,
            "return": returns,
            "gap": gap,
            "lag_gap": lag_gap,
            "gap_variance": variance,
            "gap_return_covariance": covariance,
            "beta": beta,
            "impulse": impulse,
            "noise": noise,
            "score": score,
            "true_range": tr,
            "atr": tr.rolling(params.atr_bars).mean(),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapEchoParams) -> SignalFrame:
        score = indicators["score"]
        raw = pd.Series(np.select([
            score.ge(params.min_score), score.le(-params.min_score)
        ], [1, -1], default=0), index=data.index)
        confirmed = raw.where(raw.eq(raw.shift(1)), 0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry, distance = np.nan, np.nan
        for i in range(len(data)):
            if position:
                if age == 0:
                    # Read the entry open only after the prior close's decision.
                    entry = opens[i]
                age += 1
                # Freeze ATR at the decision; mirror the fixed stop for shorts.
                if (not np.isfinite(entry) or entry <= 0
                        or not np.isfinite(closes[i])
                        or position * (closes[i] - entry) < -distance
                        or age >= params.hold_bars):
                    position, age = 0, 0
                signals[i] = position
                continue
            if (confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(closes[i]) and closes[i] > 0):
                position = int(confirmed[i])
                distance = params.stop_atr * atr[i]
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        data = pd.DataFrame({
            "open": [100, 100] + [100 + 10 * direction] * 4,
            "close": [100, 100, 100 + 9 * direction, 100 + 8 * direction,
                      100 + 6 * direction, 100 + 6 * direction],
            "high": 120.0, "low": 80.0, "volume": 1000.0,
        }, index=index)
        indicators = pd.DataFrame({"score": 0.1 * direction, "atr": 1.0}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, GapEchoParams()).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
        result = strategy.generate_signals(data, indicators, ctx, GapEchoParams(hold_bars=2)).data
        assert result["signal"].tolist()[:5] == [0, 0, direction, direction, 0]
        indicators["score"] = [0.1 * direction, 0, 0.1 * direction, 0, 0.1 * direction, 0]
        assert strategy.generate_signals(data, indicators, ctx, GapEchoParams()).data["signal"].eq(0).all()
