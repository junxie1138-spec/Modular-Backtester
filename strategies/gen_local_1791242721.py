from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    frac_window: int = 32
    frac_order: float = 0.45
    z_window: int = 40
    regime_window: int = 20
    entry_z: float = 1.25
    regime_margin: float = 0.20
    profit_target: float = 0.006
    max_hold: int = 2

    def __post_init__(self):
        if min(self.frac_window, self.z_window, self.regime_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError("frac_order must be between zero and one.")
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive.")
        if not 0.0 < self.regime_margin < 1.0:
            raise ValueError("regime_margin must be between zero and one.")
        if not 0.0 < self.profit_target < 1.0 or self.max_hold not in (1, 2):
            raise ValueError("Use a positive target below one and a one- or two-bar hold.")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791242721"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params):
        return int(params.frac_window + max(params.z_window, params.regime_window) + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.ones(params.frac_window, dtype=float)
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        # Fixed-width causal fractional differencing; no future prices.
        out["fractional_close"] = sum(
            weights[k] * close.shift(k) for k in range(params.frac_window)
        )
        history = out["fractional_close"].shift(1)
        out["fd_mean"] = history.rolling(params.z_window).mean()
        out["fd_std"] = history.rolling(params.z_window).std(ddof=0)
        out["fd_z"] = (
            (out["fractional_close"] - out["fd_mean"])
            / out["fd_std"].replace(0.0, np.nan)
        )
        out["overnight_return"] = data["open"] / close.shift(1) - 1.0
        out["intraday_return"] = close / data["open"] - 1.0
        out["overnight_energy"] = (
            out["overnight_return"].pow(2).rolling(params.regime_window).mean().shift(1)
        )
        out["intraday_energy"] = (
            out["intraday_return"].pow(2).rolling(params.regime_window).mean().shift(1)
        )
        total = out["intraday_energy"] + out["overnight_energy"]
        out["regime_balance"] = (
            (out["intraday_energy"] - out["overnight_energy"])
            / total.replace(0.0, np.nan)
        )
        out["conviction"] = (
            out["fd_z"].abs() / 3.0 * out["regime_balance"].abs()
        ).clip(0.15, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.ones(n, dtype=float)
        z = indicators["fd_z"].to_numpy()
        balance = indicators["regime_balance"].to_numpy()
        conviction = indicators["conviction"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        side, held = 0, 0
        entry_price, allocation = np.nan, 1.0
        warmup = self.warmup_bars(params)

        for i in range(n):
            if side:
                # An entry decided at the previous close fills at this open.
                if held == 0:
                    entry_price = float(opens[i])
                held += 1
                gain = side * (closes[i] / entry_price - 1.0)
                if (
                    not np.isfinite(gain)
                    or gain >= params.profit_target
                    or held >= params.max_hold
                ):
                    side, held = 0, 0
                    # Keep this decision flat; never reenter on an exit close.
                    continue
            elif (
                i >= warmup
                and np.isfinite(z[i])
                and np.isfinite(balance[i])
                and np.isfinite(conviction[i])
                and abs(z[i]) >= params.entry_z
                and abs(balance[i]) >= params.regime_margin
            ):
                # Intraday dominance follows excursions; gap dominance fades them.
                side = int(np.sign(z[i]) * np.sign(balance[i]))
                allocation = float(conviction[i])
                held = 0
            raw_signal[i] = side
            if side:
                raw_size[i] = allocation

        df = pd.DataFrame({"signal": raw_signal, "size": raw_size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    w = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=w + 6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"fd_z": np.nan, "regime_balance": np.nan, "conviction": np.nan}, index=index
    )
    indicators.iloc[w] = [2.0, 0.6, 0.4]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=w)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[w:w + 4].tolist() == [0, 1, 1, 0]
    assert result["size"].iloc[w + 1] == 0.4
    for sign in (1, -1):
        indicators.iloc[w] = [2.0, 0.6 * sign, 0.4]
        data.loc[index[w + 1], "close"] = 100.0 * (1.0 + sign * 0.01)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[w:w + 3].tolist() == [0, sign, 0]
        assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
