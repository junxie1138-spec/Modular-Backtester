from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WickParams:
    signature_bars: int = 4
    percentile_bars: int = 126
    entry_quantile: float = 0.8
    regime_bars: int = 10
    drift_bars: int = 3
    atr_bars: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 0.9
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WickParams]):
    strategy_id = "gen_local_1791314741"

    @classmethod
    def params_type(cls):
        return WickParams

    def warmup_bars(self, params: WickParams) -> int:
        return int(max(params.signature_bars + params.percentile_bars + 1,
                       params.regime_bars + params.percentile_bars + 1,
                       params.atr_bars + 1, params.drift_bars + 1))

    def indicators(self, data: pd.DataFrame, params: WickParams) -> pd.DataFrame:
        lengths = (params.signature_bars, params.percentile_bars,
                   params.regime_bars, params.drift_bars, params.atr_bars)
        if any(v < 2 for v in lengths) or not 0.5 < params.entry_quantile < 1.0:
            raise ValueError("Lookbacks must be >= 2 and entry_quantile in (0.5, 1).")
        if not 1 <= params.hold_bars <= 2 or params.trail_atr <= 0 or params.breakeven_pct <= 0:
            raise ValueError("Use 1-2 holding bars and positive stop parameters.")
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).where(lambda s: s > 0)
        top = data[["open", "close"]].max(axis=1)
        bottom = data[["open", "close"]].min(axis=1)
        out["upper_wick"] = ((data["high"] - top) / span).clip(0, 1)
        out["lower_wick"] = ((bottom - data["low"]) / span).clip(0, 1)
        upper, lower = out["upper_wick"], out["lower_wick"]
        area = pd.Series(0.0, index=data.index)
        # Second-order antisymmetric signature of cumulative wick increments.
        for lag in range(1, params.signature_bars):
            cross = upper.shift(lag) * lower - lower.shift(lag) * upper
            area = area + cross.rolling(params.signature_bars - lag).sum()
        scale = upper.rolling(params.signature_bars).sum() * lower.rolling(params.signature_bars).sum()
        out["area"] = 0.5 * area / scale.where(scale > 0)
        out["area_threshold"] = out["area"].shift(1).rolling(params.percentile_bars).quantile(params.entry_quantile)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["volatility"] = (out["true_range"] / previous.where(previous > 0)).rolling(params.regime_bars).mean()
        out["regime_threshold"] = out["volatility"].shift(1).rolling(params.percentile_bars).quantile(0.5)
        out["drift"] = data["close"].pct_change(params.drift_bars, fill_method=None)
        out["body"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1) & (out["atr"] > 0)
        stressed = out["volatility"] > out["regime_threshold"]
        regime_entry = ((~stressed) & (out["drift"] > 0)) | (stressed & (out["drift"] < 0))
        out["entry"] = (valid & regime_entry & (out["body"] > 0) &
                        (out["area"] > out["area_threshold"]) & (out["area"] > 0))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WickParams) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(n):
            op, hi, lo, close = prices[i]
            if pending:
                pending = False
                if np.isfinite(op) and op > 0 and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    entry = peak = op
                    stop = entry - params.trail_atr * atr[i - 1]
                    active, armed, age = True, False, 0
            if active:
                age += 1
                valid_bar = np.isfinite(prices[i]).all() and np.isfinite(atr[i]) and atr[i] > 0
                # Test the previously established stop before raising it.
                exit_now = not valid_bar or lo <= stop
                if not exit_now:
                    peak = max(peak, hi)
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    # Updated stop is tested at the close: intrabar ordering is unknown.
                    exit_now = close <= stop or age >= params.hold_bars
                if exit_now:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(prices[i]).all():
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": np.full(n, 0.95)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = WickParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], ["high", "close"]] = [100.8, 100.0]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
    data.loc[index[1], ["high", "close"]] = [102.0, 100.8]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
