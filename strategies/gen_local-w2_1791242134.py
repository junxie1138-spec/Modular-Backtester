from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_days: int = 3
    risk_days: int = 20
    acceleration_threshold: float = 1.0
    target_daily_vol: float = 0.008
    breakeven_pct: float = 0.006
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791242134"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return params.roc_days + params.risk_days + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.roc_days < 1 or params.risk_days < 2
                or params.acceleration_threshold <= 0
                or params.target_daily_vol <= 0
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Lookbacks and risk parameters must be positive.")
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None)
        roc = close.pct_change(params.roc_days, fill_method=None)
        acceleration = roc.diff()
        acceleration_scale = acceleration.shift(1).rolling(params.risk_days).std(ddof=0)
        score = acceleration / acceleration_scale.where(acceleration_scale > 0)
        volatility = returns.rolling(params.risk_days).std(ddof=0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.risk_days).mean()
        valid = (score.notna() & volatility.gt(0) & atr.gt(0)
                 & roc.notna() & roc.shift(1).notna())
        contracting = volatility.lt(volatility.shift(2))
        long_entry = valid & contracting & roc.gt(0) & roc.shift(1).le(0) & score.ge(params.acceleration_threshold)
        short_entry = valid & contracting & roc.lt(0) & roc.shift(1).ge(0) & score.le(-params.acceleration_threshold)
        direction = long_entry.astype(int) - short_entry.astype(int)
        conviction = (score.abs() / (2.0 * params.acceleration_threshold)).clip(0, 1)
        size = (params.target_daily_vol / volatility.where(volatility > 0) * conviction).clip(0.05, 0.95)
        return pd.DataFrame({
            "returns": returns, "roc": roc, "roc_lag": roc.shift(1),
            "acceleration": acceleration,
            "acceleration_scale": acceleration_scale, "score": score,
            "volatility": volatility, "volatility_lag": volatility.shift(2),
            "previous_close": previous_close, "true_range": true_range,
            "atr": atr, "valid": valid, "contracting": contracting,
            "long_entry": long_entry, "short_entry": short_entry,
            "conviction": conviction, "direction": direction, "entry_size": size
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        direction = indicators["direction"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        position = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        age = 0
        held_size = 1.0
        entry_atr = np.nan
        for i in range(n):
            if position:
                op, high, low, close = prices[i]
                if not np.isfinite(prices[i]).all() or op <= 0 or close <= 0:
                    position = 0
                    continue
                if np.isnan(entry):
                    entry = op
                    stop = -params.trail_atr * entry_atr
                age += 1
                # Signed price gains let both long and short stops ratchet upward.
                adverse = position * ((low if position == 1 else high) - entry)
                if adverse <= stop or age >= 2 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    position = 0
                    continue
                gain = position * (close - entry)
                if gain >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, 0.0, gain - params.trail_atr * atr[i])
                # A raised stop applies from the following bar; no assumed intrabar ordering.
                signals[i] = position
                sizes[i] = held_size
            elif direction[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(entry_sizes[i]):
                position = int(direction[i])
                entry = np.nan
                entry_atr = atr[i]
                stop = -np.inf
                armed = False
                age = 0
                held_size = float(np.clip(entry_sizes[i], 0.05, 0.95))
                signals[i] = position
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        # ponytail: daily stop detection exits at the next open; intraday fills need an execution model.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=4)
    for side in (-1, 1):
        data = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                             "close": 100.0, "volume": 1000.0}, index=index)
        features = pd.DataFrame({"direction": [side, 0, 0, 0],
                                 "atr": 2.0, "entry_size": 0.4}, index=index)
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, side, side, 0]
        assert result["size"].iloc[1:3].eq(0.4).all()
        data.loc[index[1], "low" if side == 1 else "high"] = 97.0 if side == 1 else 103.0
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, side, 0, 0]


if __name__ == "__main__":
    _self_check()
