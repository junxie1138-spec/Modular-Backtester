from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 7
    regime_window: int = 40
    pressure_window: int = 3
    atr_window: int = 14
    innovation_threshold: float = 1.1
    persistence_threshold: float = 0.15
    pressure_threshold: float = 0.12
    breakeven_pct: float = 0.008
    trail_atr: float = 1.8
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790905190"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.slope_window + params.regime_window + 3,
                       params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.slope_window < 3 or params.regime_window < 3
                or params.pressure_window < 1 or params.atr_window < 1
                or not 3 <= params.max_hold <= 5
                or params.innovation_threshold <= 0
                or not 0 < params.persistence_threshold < 1
                or not 0 < params.pressure_threshold < 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        n = params.slope_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        # Rolling local-linear latent trend estimate, oldest observation first.
        out["slope"] = sum(weights[j] * log_close.shift(n - 1 - j)
                           for j in range(n)) / float(np.square(weights).sum())
        out["innovation"] = out["slope"].diff()
        past = out["innovation"].shift(1)
        out["innovation_mean"] = past.rolling(params.regime_window).mean()
        out["innovation_std"] = past.rolling(params.regime_window).std(ddof=0)
        out["innovation_z"] = ((out["innovation"] - out["innovation_mean"])
                               / out["innovation_std"].where(out["innovation_std"] > 1e-12))
        # Use only earlier innovations to classify the current innovation.
        out["persistence"] = out["innovation"].rolling(params.regime_window).corr(past).shift(1)
        span = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["body_pressure"] = ((data["close"] - data["open"]) / span).clip(-1, 1)
        out["pressure"] = out["body_pressure"].rolling(params.pressure_window).mean()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["regime"] = np.select([
            out["persistence"] > params.persistence_threshold,
            out["persistence"] < -params.persistence_threshold,
        ], [1, -1], default=0)
        proposed = np.sign(out["innovation_z"]) * out["regime"]
        ready = (out[["innovation_z", "persistence", "pressure", "atr"]]
                 .replace([np.inf, -np.inf], np.nan).notna().all(axis=1))
        agreement = ((out["innovation_z"].abs() >= params.innovation_threshold)
                     & (proposed * out["pressure"] >= params.pressure_threshold)
                     & (out["atr"] > 0) & ready)
        out["candidate"] = proposed.where(agreement, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        side, held, armed = 0, 0, False
        entry, stop = np.nan, np.nan
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            if not np.isfinite(price) or price <= 0 or not np.isfinite(atr) or atr <= 0:
                side, held, armed = 0, 0, False
                entry = np.nan
                continue
            if side:
                if np.isnan(entry):
                    # Yesterday's decision fills at today's open.
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    stop = side * entry - params.trail_atr * atrs[i - 1]
                held += 1
                signed_price = side * price
                # Stops are evaluated at the close; exits fill next bar.
                if signed_price <= stop or held >= params.max_hold:
                    side, held, armed = 0, 0, False
                    entry = np.nan
                else:
                    if side * (price - entry) / entry >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # The stop rises in signed-price space for both directions.
                        stop = max(stop, side * entry, signed_price - params.trail_atr * atr)
            elif candidates[i] in (-1, 1):
                side = int(candidates[i])
                held, armed = 0, False
                entry = np.nan
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    strategy, params = GeneratedStrategy(), Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    indicators = pd.DataFrame({"atr": 0.2, "candidate": [0, 1, 0, 0, 0, 0]}, index=index)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0,
                         "close": [100, 100, 102, 101.4, 100, 100], "volume": 1000.0}, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    data["close"] = 200.0 - data["close"]
    indicators["candidate"] *= -1
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
