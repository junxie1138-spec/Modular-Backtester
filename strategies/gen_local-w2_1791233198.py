from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    regime_window: int = 12
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791233198"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(15, params.regime_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.regime_window < 4 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("regime_window must be >= 4 and trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["return"] = close.pct_change(fill_method=None)
        out["lag_return"] = out["return"].shift(1)
        out["serial_corr"] = out["return"].rolling(params.regime_window).corr(out["lag_return"])
        out["mean"] = close.rolling(params.regime_window).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14).mean()
        out["roc"] = close.pct_change(3, fill_method=None)
        out["acceleration"] = out["roc"].diff() / (out["atr"] / close).replace(0, np.nan)
        out["previous_acceleration"] = out["acceleration"].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        accelerating = (out["acceleration"] > 0) & (out["previous_acceleration"] <= 0)
        persistent = (out["serial_corr"] > 0.1) & (out["roc"] > 0)
        elastic = (out["serial_corr"] < -0.1) & (out["roc"] < 0) & (close < out["mean"])
        ready = out[["serial_corr", "mean", "atr", "roc", "acceleration", "previous_acceleration"]].notna().all(axis=1)
        out["entry"] = (ready & (out["atr"] > 0) & accelerating & (persistent | elastic)).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry_price = stop = 0.0
        age = 0
        for i in range(len(data)):
            # Yesterday's target fills at today's open; never use tomorrow's data.
            if not active and i > 0 and raw[i - 1] == 1:
                entry_price = opens[i]
                stop = entry_price - 2.0 * atrs[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                valid = np.isfinite([entry_price, opens[i], closes[i], lows[i], atrs[i], stop]).all()
                if not valid or entry_price <= 0 or atrs[i] <= 0:
                    active = False
                    continue
                # A stop observed during this session orders an exit next open.
                # Test the previously known stop before raising it at this close.
                if opens[i] <= stop or lows[i] <= stop or age >= 4:
                    active = False
                    continue
                # Close-confirmed +0.8% arms breakeven and the closing-price trail.
                if closes[i] >= entry_price * 1.008:
                    armed = True
                if armed:
                    stop = max(stop, entry_price, closes[i] - params.trail_atr * atrs[i])
                raw[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100, 100, 103, 103, 100, 100], "high": 104.0,
        "low": [99, 99.5, 102, 101, 99, 99],
        "close": [100, 103, 103, 102, 100, 100], "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "atr": [1, 1, 10, 10, 1, 1],
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    for n in range(1, 7):
        prefix = strategy.generate_signals(data.iloc[:n], ind.iloc[:n], None, params).data
        assert prefix["signal"].equals(result["signal"].iloc[:n])
    data.loc[:, ["open", "close"]] = 110.0
    data.loc[:, "low"] = 109.0
    data.loc[:, "high"] = 111.0
    ind.loc[:, "atr"] = 1.0
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data.loc[:, ["open", "close"]] = 100.0
    data.loc[:, "low"] = 99.5
    data.loc[index[1], "close"] = 101.0
    data.loc[index[2], "low"] = 99.0
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()
    assert strategy.warmup_bars(Params(regime_window=20)) == 22


if __name__ == "__main__":
    _self_check()
