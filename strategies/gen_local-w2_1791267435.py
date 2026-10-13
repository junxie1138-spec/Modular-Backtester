from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_span: int = 16
    innovation_window: int = 42
    season_window: int = 504
    min_samples: int = 21
    horizon: int = 18
    atr_window: int = 14
    ridge: float = 10.0
    entry_score: float = 0.25
    stop_atr: float = 3.0
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791267435"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_window + 1, params.slope_span + params.innovation_window + params.horizon + params.season_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if min(p.slope_span, p.innovation_window, p.season_window, p.min_samples, p.horizon, p.atr_window) < 2:
            raise ValueError("Lookbacks and sample counts must be at least two")
        if not (p.ridge > 0 and p.entry_score > 0 and p.stop_atr > 0 and 0 < p.min_size <= p.max_size <= 1):
            raise ValueError("Invalid regularization, thresholds, or sizing bounds")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["slope"] = out["return"].ewm(span=p.slope_span, adjust=False, min_periods=p.slope_span).mean()
        out["slope_update"] = out["slope"].diff()
        out["update_scale"] = out["slope_update"].rolling(p.innovation_window).std(ddof=0)
        out["innovation"] = (out["slope_update"] / out["update_scale"].replace(0, np.nan)).clip(-4, 4)
        out["volatility"] = out["return"].rolling(p.innovation_window).std(ddof=0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        out["month"] = data.index.month.astype(int)
        out["origin_month"] = out["month"].shift(p.horizon)
        out["mature_x"] = out["innovation"].shift(p.horizon)
        out["mature_y"] = out["log_close"].diff(p.horizon) / (
            out["volatility"].shift(p.horizon).replace(0, np.nan) * np.sqrt(p.horizon)
        )
        out["mature_y"] = out["mature_y"].replace([np.inf, -np.inf], np.nan)
        valid = out["mature_x"].notna() & out["mature_y"].notna()
        out["season_beta"] = 0.0
        for month in range(1, 13):
            belongs = valid & out["origin_month"].eq(month)
            out[f"count_{month}"] = belongs.astype(float).rolling(p.season_window, min_periods=1).sum()
            out[f"cross_{month}"] = (out["mature_x"] * out["mature_y"]).where(belongs, 0.0).rolling(p.season_window, min_periods=1).sum()
            out[f"energy_{month}"] = out["mature_x"].pow(2).where(belongs, 0.0).rolling(p.season_window, min_periods=1).sum()
            out[f"beta_{month}"] = (
                out[f"cross_{month}"] / (out[f"energy_{month}"] + p.ridge)
            ).where(out[f"count_{month}"] >= p.min_samples, 0.0)
            out["season_beta"] = out["season_beta"].where(
                out["month"].ne(month), out[f"beta_{month}"]
            )
        out["score"] = (out["season_beta"] * out["innovation"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        p = params
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.full(n, p.min_size, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        position = 0
        age = 0
        high_water = -np.inf
        low_water = np.inf
        high_stop = -np.inf
        short_stop = np.inf
        held_size = p.min_size
        warmup = self.warmup_bars(p)
        for i in range(n):
            close, atr, score = closes[i], atrs[i], scores[i]
            usable = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0 and np.isfinite(score)
            if position:
                age += 1
                if usable:
                    high_water = max(high_water, close)
                    low_water = min(low_water, close)
                    high_stop = max(high_stop, high_water - p.stop_atr * atr)
                    short_stop = min(short_stop, low_water + p.stop_atr * atr)
                # Both directions obey the required rolling-high exit.
                # Shorts also have a mirrored adverse-move trailing stop.
                if not usable or age >= p.horizon or close <= high_stop or (position < 0 and close >= short_stop):
                    position = 0
            elif usable and i >= warmup - 1 and abs(score) >= p.entry_score:
                position = 1 if score > 0 else -1
                held_size = float(np.clip(p.min_size + (p.max_size - p.min_size) * (abs(score) - p.entry_score) / p.entry_score, p.min_size, p.max_size))
                age = 0
                # Initialize watermarks on the next bar, when entry fills.
                high_water, low_water = -np.inf, np.inf
                high_stop, short_stop = -np.inf, np.inf
            signal[i] = position
            sizes[i] = held_size if position else p.min_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(p.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = Params()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.bdate_range("2015-01-02", periods=900)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index))))
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000000.0}, index=index)
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:800], p)
    pd.testing.assert_frame_equal(full.iloc[:800], prefix)
    synthetic = pd.DataFrame({"atr": 1.0, "score": 0.0}, index=index)
    entry = strategy.warmup_bars(p) - 1
    synthetic.iloc[entry, synthetic.columns.get_loc("score")] = 2 * p.entry_score
    data["close"] = 100.0
    data.iloc[entry + 2, data.columns.get_loc("close")] = 110.0
    data.iloc[entry + 3, data.columns.get_loc("close")] = 106.0
    result = strategy.generate_signals(data, synthetic, None, p).data
    assert result["signal"].iloc[entry] == 0
    assert result["signal"].iloc[entry + 1] == 1
    assert result["signal"].iloc[entry + 3] == 1
    assert result["signal"].iloc[entry + 4] == 0
    assert result["size"].iloc[entry + 1] == p.max_size
    assert result["size"].gt(0).all()
    synthetic.iloc[entry, synthetic.columns.get_loc("score")] = -2 * p.entry_score
    data["close"] = 100.0
    result = strategy.generate_signals(data, synthetic, None, p).data
    assert result["signal"].iloc[entry + 1] == -1
    assert result["signal"].iloc[entry + p.horizon] == -1
    assert result["signal"].iloc[entry + p.horizon + 1] == 0


if __name__ == "__main__":
    _self_check()
