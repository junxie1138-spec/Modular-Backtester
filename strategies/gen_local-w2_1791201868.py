from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoilParams:
    lookback: int = 60
    gap_z: float = 1.0
    min_rebound: float = 0.3
    atr_period: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 10

    def __post_init__(self):
        if self.lookback < 3 or self.atr_period < 1 or self.max_hold < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.gap_z, self.min_rebound, self.breakeven_pct, self.trail_atr)):
            raise ValueError("Thresholds and stop distances must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GapRecoilParams]):
    strategy_id = "gen_local-w2_1791201868"

    @classmethod
    def params_type(cls):
        return GapRecoilParams

    def warmup_bars(self, params: GapRecoilParams) -> int:
        return int(max(200, params.lookback + 3, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GapRecoilParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior = data["close"].shift(1).replace(0, np.nan)
        gap = (data["open"] / prior - 1).replace([np.inf, -np.inf], np.nan)
        lag = gap.shift(1)
        window = gap.rolling(params.lookback, min_periods=params.lookback)
        variance = lag.rolling(params.lookback, min_periods=params.lookback).var()
        # Fit only through yesterday; today's gap is the out-of-sample shock.
        phi = (window.cov(lag) / variance.replace(0, np.nan)).shift(1)
        mean = window.mean().shift(1)
        sigma = window.std().shift(1).replace(0, np.nan)
        out["gap"] = gap
        out["gap_lag"] = lag
        out["gap_mean"] = mean
        out["gap_sigma"] = sigma
        out["phi"] = phi
        out["gap_z"] = (gap - mean) / sigma
        # Five-session conditional rebound, excluding unconditional drift.
        out["rebound_z"] = out["gap_z"] * sum(phi ** h for h in range(1, 6))
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior).abs(),
            (data["low"] - prior).abs(),
        ], axis=1).max(axis=1)
        out["tr"] = tr
        out["atr"] = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRecoilParams) -> SignalFrame:
        eligible = (
            (data["close"] > indicators["ma200"])
            & (indicators["phi"] > -0.9)
            & (indicators["phi"] < -0.15)
            & (indicators["gap"] < 0)
            & (indicators["gap_z"] <= -params.gap_z)
            & (indicators["rebound_z"] >= params.min_rebound)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = peak = stop = 0.0
        entry_bar = -1
        for i in range(len(data)):
            o, h, l, c = prices[i]
            valid = np.isfinite(prices[i]).all() and min(o, h, l, c) > 0
            if pending:
                pending = False
                if valid and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    # Yesterday's decision fills at today's open.
                    entry = peak = o
                    stop = entry - params.trail_atr * atr[i - 1]
                    entry_bar = i
                    active = True
                    armed = False
            if active:
                if not valid or not np.isfinite(atr[i]) or atr[i] <= 0 or not np.isfinite(ma[i]):
                    active = False
                elif l <= stop:
                    # Test the carried stop before using this bar's high.
                    active = False
                else:
                    peak = max(peak, h)
                    armed = armed or peak >= entry * (1 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    active = not (
                        c <= stop or c <= ma[i]
                        or i - entry_bar + 1 >= params.max_hold
                    )
                raw[i] = int(active)
                # An exit cannot also create a fresh entry on the same close.
            elif valid and eligible[i]:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        # Stops are close-time exit decisions; execution is next open.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = GapRecoilParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[1], ["high", "low", "close"]] = [102.0, 99.0, 101.0]
    data.loc[index[2], ["high", "low", "close"]] = [104.0, 101.0, 103.0]
    data.loc[index[3], ["high", "low", "close"]] = [105.0, 101.0, 104.0]
    ind = pd.DataFrame(
        {"ma200": 90.0, "phi": -0.5, "gap": 0.0, "gap_z": 0.0,
         "rebound_z": 0.0, "atr": 1.0}, index=index,
    )
    ind.loc[index[0], ["gap", "gap_z", "rebound_z"]] = [-0.01, -2.0, 0.6875]
    # Expanding ATR must not loosen the stop ratcheted on bar two.
    ind.loc[index[3], "atr"] = 3.0
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 8
    assert (result["size"] > 0).all()
    ind["ma200"] = 200.0
    assert strategy.generate_signals(data, ind, None, p).data["signal"].eq(0).all()
    ind["ma200"] = 90.0
    flat = data.copy()
    flat[["open", "close"]] = 100.0
    flat["high"] = 100.5
    flat["low"] = 99.5
    expiry = strategy.generate_signals(flat, ind, None, p).data["signal"]
    assert expiry.tolist() == [0] + [1] * 10 + [0]
    t = np.arange(260, dtype=float)
    close = pd.Series(100 + t * 0.05 + np.sin(t), index=pd.date_range("2020-01-01", periods=260))
    history = pd.DataFrame({"open": close + 0.1, "high": close + 1,
                            "low": close - 1, "close": close, "volume": 1000.0})
    full = strategy.indicators(history, p)
    assert full.iloc[:230].equals(strategy.indicators(history.iloc[:230], p))
    assert strategy.warmup_bars(p) >= 200
    assert strategy.generate_signals(history, full, None, p).data["signal"].iloc[:200].eq(0).all()


if __name__ == "__main__":
    _self_check()
