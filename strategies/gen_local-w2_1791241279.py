from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 12
    confirmation: int = 2
    change_threshold: float = 1.2
    drift_threshold: float = 0.5
    atr_window: int = 14
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    max_hold: int = 10
    cooldown: int = 4
    min_size: float = 0.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791241279"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(2 * params.window + params.confirmation + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.window < 2 or params.confirmation < 1 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive; window must be at least two.")
        if params.change_threshold <= 0 or params.drift_threshold <= 0:
            raise ValueError("Signal thresholds must be positive.")
        close = data["close"]
        returns = np.log(close.where(close > 0)).diff()
        recent = returns.rolling(params.window, min_periods=params.window)
        recent_var = recent.var(ddof=0)
        prior_var = recent_var.shift(params.window)
        pooled_var = (recent_var + prior_var) / 2.0
        # Equal-length, disjoint windows: Gaussian variance-change likelihood gain.
        change = params.window * np.log(
            pooled_var / np.sqrt(recent_var * prior_var).replace(0.0, np.nan)
        )
        drift = recent.mean() / np.sqrt(recent_var / params.window).replace(0.0, np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = (
            (recent_var > 0) & (prior_var > 0) & (atr > 0)
            & np.isfinite(change) & np.isfinite(drift) & np.isfinite(atr)
        )
        eligible = (
            ready & (recent_var < prior_var)
            & (change >= params.change_threshold)
            & (drift >= params.drift_threshold)
        )
        confirmed = (
            eligible.astype(int).rolling(
                params.confirmation, min_periods=params.confirmation
            ).sum() == params.confirmation
        )
        entry = confirmed & ~confirmed.shift(1, fill_value=False)
        strength = (
            (change / params.change_threshold + drift / params.drift_threshold) / 4.0
        ).clip(0.0, 1.0)
        out = pd.DataFrame(
            {"recent_var": recent_var, "prior_var": prior_var,
             "pooled_var": pooled_var, "change": change, "drift": drift,
             "true_range": true_range, "atr": atr, "ready": ready,
             "eligible": eligible, "confirmed": confirmed, "entry": entry,
             "strength": strength},
            index=data.index,
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        if (
            params.max_hold < 1 or params.cooldown < 0
            or params.initial_stop_atr <= 0 or params.trail_atr <= 0
            or params.breakeven_pct <= 0
            or not 0 < params.min_size <= params.max_size <= 1
        ):
            raise ValueError("Invalid holding, stop, or sizing parameters.")
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        strength = indicators["strength"].to_numpy(dtype=float)
        active = pending = armed = False
        blocked_until = -1
        held = 0
        entry_price = stop = peak = entry_atr = 0.0
        exposure = params.min_size
        for i in range(count):
            # Yesterday's entry decision fills at today's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = peak = opens[i]
                    stop = entry_price - params.initial_stop_atr * entry_atr
                    held = 0
                    armed = False
                else:
                    blocked_until = i + params.cooldown
            if active:
                held += 1
                price = closes[i]
                # Stops are evaluated at the close; the flat target fills next bar.
                if (
                    not np.isfinite(price) or price <= 0
                    or not np.isfinite(atr[i]) or atr[i] <= 0
                    or price <= stop or held >= params.max_hold
                ):
                    active = False
                    blocked_until = i + params.cooldown
                else:
                    peak = max(peak, price)
                    armed = armed or price >= entry_price * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                    if price <= stop:
                        active = False
                        blocked_until = i + params.cooldown
                    else:
                        signals[i] = 1
                        sizes[i] = exposure
            elif entries[i] and i > blocked_until and np.isfinite(atr[i]) and atr[i] > 0:
                exposure = params.min_size + (
                    params.max_size - params.min_size
                ) * float(np.clip(strength[i], 0.0, 1.0))
                entry_atr = atr[i]
                pending = True
                signals[i] = 1
                sizes[i] = exposure
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": [99., 100., 102., 104., 101., 101., 101.],
         "high": [100., 103., 105., 105., 102., 102., 102.],
         "low": [98., 99., 101., 100., 100., 100., 100.],
         "close": [99., 102., 104., 101., 101., 101., 101.],
         "volume": 1000.},
        index=index,
    )
    indicators = pd.DataFrame(
        {"atr": 1., "entry": [True, False, False, False, False, False, False],
         "strength": 0.8},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.8)
    assert (result["size"] > 0).all()
    calculated = GeneratedStrategy.indicators(None, data, params)
    assert calculated.index.equals(data.index)
    assert np.isfinite(calculated.select_dtypes(include="number").to_numpy()).all()
    assert not calculated["entry"].any()
    assert GeneratedStrategy.warmup_bars(None, params) >= 2 * params.window + params.confirmation + 1


if __name__ == "__main__":
    _self_check()
