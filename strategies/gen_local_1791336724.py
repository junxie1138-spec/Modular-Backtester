from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entropy_window: int = 24
    mean_window: int = 20
    atr_period: int = 14
    entropy_floor: float = 0.88
    stretch_atr: float = 1.8
    stop_atr: float = 2.2
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791336724"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.entropy_window + 3, params.mean_window + 2, params.atr_period + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.entropy_window, params.mean_window, params.atr_period, params.holding_bars) < 1:
            raise ValueError("Window lengths and holding_bars must be positive.")
        if not (0.0 <= params.entropy_floor <= 1.0 and params.stretch_atr > 0.0 and params.stop_atr > 0.0):
            raise ValueError("Invalid entropy threshold, stretch, or stop.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        a, b, c = out["return_lag2"], out["return_lag1"], out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns are ordered by age, giving six consistent ordinal patterns.
        out["ordinal_code"] = ((a <= b).astype(int) + 2 * (a <= c).astype(int) + 4 * (b <= c).astype(int)).where(valid)
        terms = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f"pattern_probability_{code}"
            out[name] = out["ordinal_code"].eq(code).astype(float).where(valid).rolling(params.entropy_window, min_periods=params.entropy_window).mean()
            probability = out[name]
            terms.append(-probability * np.log(probability.clip(lower=1e-12)))
        out["permutation_entropy"] = pd.concat(terms, axis=1).sum(axis=1, min_count=6) / np.log(6.0)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(data[["high", "low", "close"]].notna().all(axis=1))
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["center"] = out["previous_close"].rolling(params.mean_window, min_periods=params.mean_window).mean()
        out["stretch"] = ((data["close"] - out["center"]) / out["atr"].where(out["atr"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["previous_stretch"] = out["stretch"].shift(1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        entropy = indicators["permutation_entropy"]
        stretch = indicators["stretch"]
        previous = indicators["previous_stretch"]
        disorder = entropy.ge(params.entropy_floor)
        long_entry = (disorder & stretch.le(-params.stretch_atr) & stretch.gt(previous)).to_numpy()
        short_entry = (disorder & stretch.ge(params.stretch_atr) & stretch.lt(previous)).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry_price, entry_atr = np.nan, np.nan
        for i in range(len(data)):
            if position:
                # The previous decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                age += 1
                invalid = not (np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]))
                stopped = position * (closes[i] - entry_price) <= -params.stop_atr * entry_atr
                if invalid or stopped or age >= params.holding_bars:
                    position = 0
            elif np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                if long_entry[i] or short_entry[i]:
                    position = 1 if long_entry[i] else -1
                    entry_price, entry_atr, age = np.nan, atr[i], 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=18)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[1], "open"] = 101.0
    data.loc[index[2], "close"] = 98.0
    data.loc[index[4], "close"] = 103.0
    features = pd.DataFrame({"permutation_entropy": 1.0, "stretch": 0.0, "previous_stretch": 0.0, "atr": 1.0}, index=index)
    features.loc[index[[0, 6]], ["stretch", "previous_stretch"]] = [-2.0, -3.0]
    features.loc[index[3], ["stretch", "previous_stretch"]] = [2.0, 3.0]
    features.loc[index[2], "atr"] = 100.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0, 0] + [1] * 8 + [0] * 3
    assert result["size"].gt(0).all()
    index = pd.date_range("2020-01-01", periods=80)
    close = 100.0 + 2.0 * np.sin(np.arange(80) * 1.7)
    history = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(history, Params())
    prefix = strategy.indicators(history.iloc[:50], Params())
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)
    assert full["permutation_entropy"].dropna().between(-1e-12, 1.0 + 1e-12).all()
    assert full["permutation_entropy"].first_valid_index() == index[26]


if __name__ == "__main__":
    _self_check()
