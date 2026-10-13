from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    path_window: int = 12
    peak_window: int = 60
    atr_window: int = 14
    recovery_window: int = 3
    min_drawdown: float = 0.03
    max_drawdown: float = 0.15
    area_threshold: float = 0.025
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790893804"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.peak_window, params.path_window + 1,
                       params.atr_window + 1, params.recovery_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.path_window, params.peak_window, params.atr_window,
               params.recovery_window, params.max_hold) < 2:
            raise ValueError("Lookbacks and max_hold must be at least two.")
        if not (0 < params.min_drawdown < params.max_drawdown < 1
                and params.area_threshold > 0 and params.breakeven_pct > 0
                and params.trail_atr > 0):
            raise ValueError("Invalid thresholds.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["overnight"] = np.log(opening / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = np.log(close / opening).replace([np.inf, -np.inf], np.nan)
        out["x"] = out["overnight"].fillna(0).cumsum()
        out["y"] = out["intraday"].fillna(0).cumsum()
        out["cross"] = (out["x"].shift(1) * out["intraday"]
                        - out["y"].shift(1) * out["overnight"])
        w = params.path_window
        out["overnight_sum"] = out["overnight"].rolling(w).sum()
        out["intraday_sum"] = out["intraday"].rolling(w).sum()
        out["path_length"] = (out["overnight"].abs() + out["intraday"].abs()).rolling(w).sum()
        # Translation-invariant antisymmetric second-order signature area.
        out["area"] = 0.5 * (
            out["cross"].rolling(w).sum()
            - out["x"].shift(w) * out["intraday_sum"]
            + out["y"].shift(w) * out["overnight_sum"]
        ) / out["path_length"].pow(2).replace(0, np.nan)
        out["peak"] = close.rolling(params.peak_window).max()
        out["drawdown"] = 1 - close / out["peak"]
        out["recovery"] = close / close.shift(params.recovery_window) - 1
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        eligible = (
            out["drawdown"].between(params.min_drawdown, params.max_drawdown)
            & out["overnight_sum"].lt(0)
            & out["intraday_sum"].gt(0)
            & out["atr"].gt(0)
        )
        out["long_setup"] = (
            eligible & out["area"].lt(-params.area_threshold)
            & out["recovery"].gt(0)
        ).fillna(False)
        out["short_setup"] = (
            eligible & out["area"].gt(params.area_threshold)
            & out["recovery"].lt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        long_confirmed = (indicators["long_setup"]
                          & indicators["long_setup"].shift(1, fill_value=False))
        short_confirmed = (indicators["short_setup"]
                           & indicators["short_setup"].shift(1, fill_value=False))
        candidates = np.where(long_confirmed, 1, np.where(short_confirmed, -1, 0))
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry, stop = np.nan, np.nan
        armed = False
        for i in range(len(data)):
            valid = (np.isfinite(close[i]) and close[i] > 0
                     and np.isfinite(atr[i]) and atr[i] > 0)
            if position:
                # A prior close's entry decision fills at this bar's open.
                if np.isnan(entry):
                    if not np.isfinite(opening[i]) or opening[i] <= 0 or not valid:
                        position = 0
                        continue
                    entry = opening[i]
                    stop = position * entry - params.trail_atr * atr[i - 1]
                age += 1
                if not valid:
                    position = 0
                    continue
                favorable_price = position * close[i]
                signed_entry = position * entry
                # Both directions ratchet upward in signed-price space.
                # For shorts this makes the actual buy-stop price decrease.
                if favorable_price - signed_entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, signed_entry,
                               favorable_price - params.trail_atr * atr[i])
                # Stops are evaluated at the close; exits fill next bar.
                if favorable_price <= stop or age >= params.max_hold:
                    position = 0
                    continue
            elif valid and candidates[i]:
                position = int(candidates[i])
                entry, stop = np.nan, np.nan
                age, armed = 0, False
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        close = 100 + direction * np.array([0, 0, 3, 5, 3, 2, 2], dtype=float)
        data = pd.DataFrame({
            "open": 100.0, "high": np.maximum(100, close) + 1,
            "low": np.minimum(100, close) - 1,
            "close": close, "volume": 1000.0,
        }, index=index)
        ind = pd.DataFrame({"atr": 1.0, "long_setup": False,
                            "short_setup": False}, index=index)
        column = "long_setup" if direction == 1 else "short_setup"
        ind.loc[index[:2], column] = True
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0, 0]
        ind.loc[index[1], column] = False
        assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    index = pd.date_range("2020-01-01", periods=160)
    close = 100 * np.exp(0.06 * np.sin(np.arange(160) / 9))
    data = pd.DataFrame({"open": close * 1.001, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1000.0}, index=index)
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    prefix_data = data.iloc[:120]
    prefix = strategy.generate_signals(prefix_data, strategy.indicators(prefix_data, params), ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:120], prefix)
    assert result["signal"].isin([-1, 0, 1]).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    self_check()
