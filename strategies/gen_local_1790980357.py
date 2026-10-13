from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    lookback: int = 60
    atr_period: int = 14
    hold_days: int = 4
    threshold: float = 0.25
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1790980357"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(params.lookback + 3, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if not (params.lookback >= 5 and params.atr_period >= 2
                and 3 <= params.hold_days <= 5 and params.threshold > 0
                and params.breakeven_pct > 0 and params.trail_atr > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        r = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        x1, x2 = r.shift(1), r.shift(2)
        window = params.lookback
        v1, v2 = x1.rolling(window).var(), x2.rolling(window).var()
        c12 = x1.rolling(window).cov(x2)
        c1, c2 = r.rolling(window).cov(x1), r.rolling(window).cov(x2)
        determinant = v1 * v2 - c12 * c12
        denominator = determinant.where(determinant > 1e-20)
        b1 = (c1 * v2 - c2 * c12) / denominator
        b2 = (c2 * v1 - c1 * c12) / denominator
        mean = r.rolling(window).mean()
        intercept = mean - b1 * x1.rolling(window).mean() - b2 * x2.rolling(window).mean()
        previous, older = r, x1
        forecast = pd.Series(0.0, index=data.index)
        for _ in range(params.hold_days):
            next_return = intercept + b1 * previous + b2 * older
            forecast = forecast + next_return - mean
            older, previous = previous, next_return
        volatility = r.rolling(window).std().replace(0.0, np.nan)
        score = forecast / (volatility * np.sqrt(params.hold_days))
        # Require a stable AR(2) model whose delayed echo dominates lag one.
        stable = (b2.abs() < 1) & (b1 + b2 < 1) & (b2 - b1 < 1)
        eligible = stable & (b2.abs() >= 0.1) & (b2.abs() > b1.abs())
        previous_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["return"] = r
        out["lag_one"] = x1
        out["lag_two"] = x2
        out["beta_one"] = b1
        out["beta_two"] = b2
        out["forecast_excess"] = forecast
        out["volatility"] = volatility
        out["score"] = score.replace([np.inf, -np.inf], np.nan).where(eligible, 0.0).fillna(0.0)
        out["atr"] = tr.rolling(params.atr_period).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        side, age = 0, 0
        entry, stop, exposure = np.nan, np.nan, 1.0
        armed = False
        score = indicators["score"].to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        for i in range(n):
            if side:
                # Yesterday's decision fills at today's open.
                if np.isnan(entry):
                    entry = float(opens[i])
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    stop = side * entry - params.trail_atr * atr[i - 1]
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]):
                    side = 0
                    entry = np.nan
                    continue
                signed_close = side * closes[i]
                # Close-confirmed stops submit a flat target for the next open.
                exited = signed_close <= stop or age >= params.hold_days
                if not exited:
                    if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed profit-space stop rises for both trade directions.
                        stop = max(stop, side * entry,
                                   signed_close - params.trail_atr * atr[i])
                    exited = signed_close <= stop
                if exited:
                    side, age = 0, 0
                    entry = np.nan
                    armed = False
                    continue
            elif (i >= self.warmup_bars(params) - 1
                  and np.isfinite(atr[i]) and atr[i] > 0
                  and abs(score[i]) >= params.threshold):
                side = 1 if score[i] > 0 else -1
                entry, age, armed = np.nan, 0, False
                exposure = float(np.clip(abs(score[i]), 0.15, 0.95))
            signals[i] = side
            sizes[i] = exposure if side else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = EchoParams()
    s = GeneratedStrategy()
    rng = np.random.default_rng(7)
    r = np.zeros(240)
    for i in range(2, len(r)):
        r[i] = 0.65 * r[i - 2] + rng.normal(0, 0.008)
    close = 100 * np.cumprod(1 + r)
    opening = np.r_[100.0, close[:-1]]
    d = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 0.5,
        "low": np.minimum(opening, close) - 0.5, "close": close,
        "volume": np.full(len(close), 1000000.0)
    }, index=pd.date_range("2020-01-01", periods=len(close), freq="B"))
    result = s.generate_signals(d, s.indicators(d, p), None, p).data
    prefix = d.iloc[:180]
    check = s.generate_signals(prefix, s.indicators(prefix, p), None, p).data
    pd.testing.assert_frame_equal(result.iloc[:180], check)
    assert set(result["signal"]).issubset({-1, 0, 1})
    assert (result["size"] > 0).all()
    assert result["signal"].iloc[:s.warmup_bars(p)].eq(0).all()
    assert (result["signal"] != 0).any()
