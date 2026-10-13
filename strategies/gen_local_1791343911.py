from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 20
    corr_window: int = 40
    z_entry: float = 1.1
    corr_min: float = 0.1
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791343911"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.ma_window, params.corr_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.ma_window, params.corr_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not (np.isfinite(params.z_entry) and params.z_entry > 0
                and np.isfinite(params.corr_min) and 0 < params.corr_min < 1
                and np.isfinite(params.stop_atr) and params.stop_atr > 0
                and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid thresholds, stop distance, or holding horizon.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["return"] = close.pct_change(fill_method=None)
        out["lag1"] = out["return"].shift(1)
        out["lag2"] = out["return"].shift(2)
        out["rho1"] = out["return"].rolling(params.corr_window).corr(out["lag1"]).clip(-1.0, 1.0)
        out["rho2"] = out["return"].rolling(params.corr_window).corr(out["lag2"]).clip(-1.0, 1.0)
        # A lag-two echo forecasts tomorrow using yesterday's return.
        out["echo"] = out["rho2"] * out["lag1"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        phase = ((indicators["rho1"] <= -params.corr_min)
                 & (indicators["rho2"] >= params.corr_min))
        long_entry = phase & (indicators["z"] <= -params.z_entry) & (indicators["echo"] > 0)
        short_entry = phase & (indicators["z"] >= params.z_entry) & (indicators["echo"] < 0)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_atr = stop = np.nan
        first_ready = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # Read the fill-bar open only once that bar has arrived.
                    # Freeze decision-bar ATR; never ratchet the stop.
                    stop = opens[i] - side * params.stop_atr * entry_atr
                invalid = not (np.isfinite(stop) and np.isfinite(closes[i]) and closes[i] > 0)
                stopped = (closes[i] < stop) if side == 1 else (closes[i] > stop)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    side = 0
                raw[i] = side
                continue
            if (i >= first_ready and candidate[i] != 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                    and np.isfinite(atr[i]) and atr[i] > 0):
                side = int(candidate[i])
                entry_bar = i + 1
                entry_atr = atr[i]
                stop = np.nan
                raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GeneratedParams(ma_window=2, corr_window=2, atr_window=2)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=16, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[5], "close"] = 97.0
    data.loc[index[8], "close"] = 103.0
    ind = pd.DataFrame({"rho1": -0.5, "rho2": 0.5, "z": 0.0,
                        "echo": 0.0, "atr": 1.0}, index=index)
    ind.loc[index[[4, 10]], ["z", "echo"]] = [-2.0, 0.01]
    ind.loc[index[7], ["z", "echo"]] = [2.0, -0.01]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    expected = np.zeros(16, dtype=int)
    expected[5] = 1
    expected[8] = -1
    expected[11:15] = 1
    assert np.array_equal(signals["signal"].to_numpy(), expected)
    assert (signals["size"] > 0).all()
    # Changing ATR after entry must not widen the fixed stop.
    altered = ind.copy()
    altered.loc[index[5], "atr"] = 100.0
    assert strategy.generate_signals(data, altered, ctx, params).data.equals(signals)
    veto = ind.copy()
    veto["rho2"] = -0.5
    assert (strategy.generate_signals(data, veto, ctx, params).data["signal"] == 0).all()
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], ind.iloc[:end], ctx, params).data
        assert prefix.equals(signals.iloc[:end])
    flat = data.copy()
    flat["close"] = 100.0
    actual = strategy.indicators(flat, params)
    assert actual["z"].isna().all()
    assert (strategy.generate_signals(flat, actual, ctx, params).data["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
