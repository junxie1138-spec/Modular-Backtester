from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_window: int = 20
    volatility_window: int = 20
    atr_window: int = 14
    volume_multiple: float = 1.15
    target_volatility: float = 0.12
    breakeven_pct: float = 0.006
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791324669"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.volume_window + 2,
                       params.volatility_window + 1,
                       params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.volume_window, params.volatility_window,
               params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (params.volume_multiple, params.target_volatility,
                    params.breakeven_pct, params.trail_atr)):
            raise ValueError("Thresholds and risk parameters must be positive.")
        close = data["close"]
        previous = close.shift(1).where(close.shift(1) > 0)
        returns = close.div(previous).sub(1)
        gap = data["open"].div(previous).sub(1)
        body = close.div(data["open"].where(data["open"] > 0)).sub(1)
        volume_base = data["volume"].shift(1).rolling(
            params.volume_window, min_periods=params.volume_window).mean()
        volume_ratio = data["volume"].div(volume_base.where(volume_base > 0))
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        volatility = returns.rolling(
            params.volatility_window,
            min_periods=params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        ready = (atr > 0) & (volatility > 0) & (volume_ratio >= params.volume_multiple)
        upward = ready & (gap < 0) & (body > 0) & (returns > 0)
        downward = ready & (gap > 0) & (body < 0) & (returns < 0)
        direction = upward.astype(int) - downward.astype(int)
        confirmed = direction.where(
            direction.eq(direction.shift(1)) & direction.ne(0), 0)
        result = pd.DataFrame({
            "return": returns, "gap": gap, "body": body,
            "volume_base": volume_base, "volume_ratio": volume_ratio,
            "true_range": true_range, "atr": atr,
            "volatility": volatility, "direction": direction,
            "confirmed": confirmed,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.05, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        volatility = indicators["volatility"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        side = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        age = 0
        allocation = 0.05
        for i in range(n):
            valid = (np.isfinite(opens[i]) and np.isfinite(highs[i])
                     and np.isfinite(lows[i]) and np.isfinite(closes[i])
                     and min(opens[i], highs[i], lows[i], closes[i]) > 0
                     and lows[i] <= min(opens[i], closes[i])
                     and highs[i] >= max(opens[i], closes[i]))
            if not valid:
                side = 0
                entry = np.nan
                continue
            if side:
                # A decision on the preceding close fills at this open.
                if np.isnan(entry):
                    entry = opens[i]
                    stop = side * entry - params.trail_atr * atr[i - 1]
                    age = 0
                    armed = False
                age += 1
                # Signed prices make both long and short stops ratchet upward.
                adverse = lows[i] if side == 1 else -highs[i]
                if adverse <= stop or age >= 2:
                    side = 0
                    entry = np.nan
                    size[i] = allocation
                    continue
                favorable = highs[i] if side == 1 else -lows[i]
                if favorable - side * entry >= params.breakeven_pct * entry:
                    armed = True
                if armed:
                    # New stops become effective next session; no OHLC ordering guess.
                    stop = max(stop, side * entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, favorable - params.trail_atr * atr[i])
                signal[i] = side
                size[i] = allocation
                continue
            if (confirmed[i] in (-1, 1) and np.isfinite(atr[i])
                    and atr[i] > 0 and np.isfinite(volatility[i])
                    and volatility[i] > 0):
                side = int(confirmed[i])
                entry = np.nan
                allocation = float(np.clip(
                    params.target_volatility / volatility[i], 0.05, 0.95))
                signal[i] = side
                size[i] = allocation
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [100.2, 100.2, 101.2, 100.2, 100.2, 100.2],
        "low": [99.8] * 6,
        "close": [100.0, 100.0, 101.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    ind = pd.DataFrame({
        "confirmed": [0, 1, 0, 0, 0, 0],
        "atr": [0.5] * 6, "volatility": [0.2] * 6,
    }, index=index)
    for direction in (1, -1):
        bars = data.copy()
        if direction == -1:
            bars["open"] = 200.0 - data["open"]
            bars["close"] = 200.0 - data["close"]
            bars["high"] = 200.0 - data["low"]
            bars["low"] = 200.0 - data["high"]
        features = ind.copy()
        features["confirmed"] *= direction
        result = strategy.generate_signals(bars, features, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, 0, 0]
        assert np.isclose(result["size"].iloc[2], 0.6)
        if direction == 1:
            bars.loc[index[2], "low"] = 98.0
        else:
            bars.loc[index[2], "high"] = 102.0
        stopped = strategy.generate_signals(bars, features, None, params).data
        assert stopped["signal"].tolist() == [0, 0, direction, 0, 0, 0]
    warmup = strategy.indicators(data, params)
    assert np.isfinite(warmup.to_numpy()).all()
    assert not warmup["confirmed"].any()
    assert strategy.warmup_bars(params) >= 22


if __name__ == "__main__":
    _self_check()
