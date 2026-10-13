from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 12
    noise_window: int = 20
    efficiency_window: int = 10
    atr_window: int = 14
    innovation_threshold: float = 1.0
    efficiency_threshold: float = 0.35
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    hold_days: int = 4

    def __post_init__(self):
        if min(self.trend_window, self.noise_window, self.efficiency_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 3 <= self.hold_days <= 5:
            raise ValueError("hold_days must be between three and five.")
        if not 0 < self.efficiency_threshold < 1:
            raise ValueError("efficiency_threshold must be between zero and one.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.innovation_threshold, self.breakeven_pct, self.trail_atr)):
            raise ValueError("Threshold and stop parameters must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791311642"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.trend_window + params.noise_window + 1,
                       params.efficiency_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        log_price = np.log(close)
        n = params.trend_window
        center = (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        weighted = pd.Series(0.0, index=data.index)
        for lag in range(n):
            weighted = weighted + (center - lag) * log_price.shift(lag)
        slope = weighted / denominator
        level = log_price.rolling(n, min_periods=n).mean()
        forecast = (level + (center + 1) * slope).shift(1)
        residual = log_price - forecast
        # Updating the rolling local-linear fit revises its latent slope.
        innovation = residual * (6.0 / (n * (n + 1)))
        history = innovation.shift(1)
        noise_mean = history.rolling(params.noise_window, min_periods=params.noise_window).mean()
        noise_std = history.rolling(params.noise_window, min_periods=params.noise_window).std(ddof=0)
        innovation_z = (innovation - noise_mean) / noise_std.where(noise_std > 0)
        movement = log_price.diff().abs().rolling(
            params.efficiency_window, min_periods=params.efficiency_window).sum()
        efficiency = log_price.diff(params.efficiency_window).abs() / movement.where(movement > 0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({
            "log_price": log_price, "slope": slope, "level": level,
            "forecast": forecast, "residual": residual, "innovation": innovation,
            "noise_mean": noise_mean, "noise_std": noise_std,
            "innovation_z": innovation_z, "movement": movement,
            "efficiency": efficiency, "true_range": true_range, "atr": atr,
            "ma200": close.rolling(200, min_periods=200).mean()
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        required = indicators[["slope", "innovation_z", "efficiency", "atr", "ma200"]]
        ready = np.isfinite(required).all(axis=1) & (indicators["atr"] > 0)
        direction = pd.Series(np.where(data["close"] > indicators["ma200"], 1, -1),
                              index=data.index)
        efficient = indicators["efficiency"] >= params.efficiency_threshold
        # Persistent auctions follow slope revisions; choppy auctions fade them.
        aligned_surprise = direction * indicators["innovation_z"]
        score = aligned_surprise.where(efficient, -aligned_surprise)
        qualifies = (ready & (direction * indicators["slope"] > 0)
                     & (score >= params.innovation_threshold))
        candidate = direction.where(qualifies, 0).astype(int)
        event = candidate.where(candidate.ne(candidate.shift(1).fillna(0)), 0)
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int),
                           "size": np.ones(len(data), dtype=float)}, index=data.index)
        side = 0
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        age = 0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if side:
                if pending:
                    entry = float(data["open"].iloc[i])
                    pending = False
                    age = 0
                    armed = False
                    # Signed-price coordinates make both directions ratchet upward.
                    stop = side * entry - params.trail_atr * float(indicators["atr"].iloc[i - 1])
                age += 1
                adverse = float(data["low"].iloc[i]) if side == 1 else -float(data["high"].iloc[i])
                valid = bool(ready.iloc[i]) and np.isfinite(entry) and entry > 0
                exit_now = (not valid or adverse <= stop or age >= params.hold_days
                            or side * (close - float(indicators["ma200"].iloc[i])) <= 0)
                if exit_now:
                    side = 0
                    armed = False
                    # Stay flat for this decision bar; no same-bar re-entry.
                    continue
                if side * (close - entry) >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, side * close - params.trail_atr * atr)
                # A newly raised stop takes effect on the following bar.
            elif int(event.iloc[i]) != 0 and bool(ready.iloc[i]):
                side = int(event.iloc[i])
                pending = True
            df.iat[i, 0] = side
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    strategy = GeneratedStrategy()
    params = Params()
    for side in (1, -1):
        close = np.array([100, 101, 103, 101, 100, 100, 100], dtype=float)
        if side == -1:
            close = 200 - close
        data = pd.DataFrame({"open": 100.0, "high": close + 0.1,
                             "low": close - 0.1, "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"slope": side * 0.01,
                            "innovation_z": [side * 2.0, 0, 0, 0, 0, 0, 0],
                            "efficiency": 0.8, "atr": 1.0,
                            "ma200": 90.0 if side == 1 else 110.0}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0, 0, 0]
        assert (result["size"] > 0).all()
        flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
        timed = strategy.generate_signals(flat, ind, None, params).data
        assert timed["signal"].tolist() == [0, side, side, side, side, 0, 0]
    warm_data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                              "close": 100.0, "volume": 1000.0},
                             index=pd.date_range("2020-01-01", periods=240))
    features = strategy.indicators(warm_data, params)
    assert features.index.equals(warm_data.index)
    assert strategy.generate_signals(warm_data, features, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
