from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    edge: float = 0.35
    trail_atr: float = 1.8


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791305029"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not np.isfinite(params.edge) or params.edge <= 0:
            raise ValueError("edge must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["return"] = out["log_close"].diff()
        out["slope"] = out["return"].ewm(span=8, adjust=False, min_periods=8).mean()
        out["volatility"] = out["return"].rolling(20, min_periods=20).std().replace(0, np.nan)
        out["innovation"] = (out["return"] - out["slope"].shift(1)) / out["volatility"]
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["atr_fraction"] = (out["atr"] / close).replace(0, np.nan)
        for lag in range(4):
            out[f"shape_{lag}"] = out["innovation"].shift(lag)
        out["trend_state"] = out["slope"] / out["volatility"]
        features = out[[f"shape_{lag}" for lag in range(4)] + ["trend_state"]]
        log_open = np.log(data["open"].where(data["open"] > 0))
        distances, outcomes = [], []
        # Fixed, bounded search; every candidate's four-session outcome has matured.
        for lag in range(12, 161):
            delta = features - features.shift(lag)
            distances.append(delta.pow(2).sum(axis=1, min_count=5).to_numpy())
            outcomes.append(
                ((log_open.shift(lag - 5) - log_open.shift(lag - 1))
                 / out["atr_fraction"].shift(lag)).to_numpy()
            )
        distance = np.asarray(distances)
        outcome = np.asarray(outcomes)
        valid = np.isfinite(distance) & np.isfinite(outcome)
        distance = np.where(valid, distance, np.inf)
        nearest = np.argsort(distance, axis=0, kind="stable")[:7]
        chosen_distance = np.take_along_axis(distance, nearest, axis=0)
        chosen_outcome = np.take_along_axis(outcome, nearest, axis=0)
        ready = np.isfinite(chosen_distance).all(axis=0)
        clean = np.where(np.isfinite(chosen_outcome), chosen_outcome, 0.0)
        forecast = clean.mean(axis=0)
        out["forecast"] = np.where(ready, forecast, np.nan)
        out["agreement"] = np.where(
            ready, (np.sign(clean) == np.sign(forecast)[None, :]).mean(axis=0), np.nan
        )
        out["match_distance"] = np.where(ready, chosen_distance.mean(axis=0), np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams
    ) -> SignalFrame:
        eligible = (
            indicators["forecast"].abs().ge(params.edge)
            & indicators["agreement"].ge(5.0 / 7.0)
            & indicators["match_distance"].le(2.0)
            & indicators["atr"].gt(0)
        ).fillna(False).to_numpy()
        forecast = indicators["forecast"].fillna(0).to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        desired = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry = peak = stop = 0.0
        armed = False
        for i in range(len(data)):
            # Yesterday's decision fills at today's open.
            if side == 0 and i > 0 and desired[i - 1] != 0:
                side = int(desired[i - 1])
                entry = float(opens[i])
                peak = side * entry
                stop = peak - params.trail_atr * float(atr[i - 1])
                age, armed = 0, False
            if side != 0:
                age += 1
                favorable_close = side * float(closes[i])
                if not np.isfinite(favorable_close) or not np.isfinite(atr[i]):
                    side = 0
                else:
                    peak = max(peak, favorable_close)
                    if favorable_close - side * entry >= 0.01 * entry:
                        armed = True
                    if armed:
                        # Signed-price stop only rises; short price stops only fall.
                        stop = max(stop, side * entry, peak - params.trail_atr * atr[i])
                    if favorable_close <= stop or age >= 4:
                        side = 0
                desired[i] = side
            elif i >= self.warmup_bars(params) and eligible[i]:
                desired[i] = 1 if forecast[i] > 0 else -1
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AnalogParams()
    strategy = GeneratedStrategy()
    n = 260
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(0.002 * np.sin(np.arange(n) / 3.0) + 0.0002 * np.arange(n))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000000.0}, index=index
    )
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        full.iloc[:230], strategy.indicators(data.iloc[:230], params)
    )
    for direction in (1, -1):
        fixture = pd.DataFrame(
            {"forecast": 0.0, "agreement": 1.0, "match_distance": 0.0, "atr": 1.0},
            index=index
        )
        fixture.iloc[200, fixture.columns.get_loc("forecast")] = direction
        prices = data.copy()
        prices["open"] = 100.0
        prices["close"] = 100.0
        prices.iloc[201, prices.columns.get_loc("close")] = 100.0 + direction * 1.2
        prices.iloc[202, prices.columns.get_loc("close")] = 100.0 + direction * 3.0
        prices.iloc[203, prices.columns.get_loc("close")] = 100.0 + direction * 0.5
        signals = strategy.generate_signals(prices, fixture, None, params).data
        assert signals["signal"].iloc[200:205].tolist() == [0, direction, direction, direction, 0]
        prices["close"] = 100.0
        timed = strategy.generate_signals(prices, fixture, None, params).data
        assert timed["signal"].iloc[200:206].tolist() == [0, direction, direction, direction, direction, 0]
        assert signals["size"].gt(0).all()
        assert signals["signal"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
