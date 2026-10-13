from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    fast_window: int = 5
    slow_window: int = 20
    atr_window: int = 14
    release_ratio: float = 0.85
    target_vol: float = 0.12
    stop_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = "gen_local-w2_1791276394"

    @classmethod
    def params_type(cls):
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return int(max(params.fast_window, params.slow_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        if not (2 <= params.fast_window < params.slow_window
                and params.atr_window >= 1 and params.hold_bars in (1, 2)
                and 0 < params.release_ratio < 1
                and np.isfinite(params.target_vol) and params.target_vol > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid volatility-capacity parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        squared = out["return"].pow(2)
        out["fast_vol"] = np.sqrt(squared.rolling(params.fast_window).mean())
        out["slow_vol"] = np.sqrt(squared.rolling(params.slow_window).mean())
        previous = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        out["eligible"] = (
            (out["return"] > 0)
            & (out["fast_vol"] > 0)
            & (out["fast_vol"] < out["fast_vol"].shift(1))
            & (out["fast_vol"] < params.release_ratio * out["slow_vol"])
            & (out["atr"] > 0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        out["confirmed"] = (
            out["eligible"] & out["eligible"].shift(1, fill_value=False)
        )
        out["target_size"] = (
            params.target_vol / (np.sqrt(252.0) * out["fast_vol"].replace(0, np.nan))
        ).replace([np.inf, -np.inf], np.nan).clip(0.01, 0.95).fillna(0.01)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CapacityParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.01, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        targets = indicators["target_size"].to_numpy(dtype=float)
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        entry_size = 0.01
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The unshifted entry decision fills at this bar's open.
                    stop = opens[i] - params.stop_atr * entry_atr
                size[i] = entry_size
                if (not np.isfinite(stop) or not np.isfinite(closes[i])
                        or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif (confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(targets[i]) and targets[i] > 0):
                entry_bar = i + 1
                entry_atr = atr[i]
                entry_size = float(np.clip(targets[i], 0.01, 0.95))
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "atr": 2.0, "confirmed": [False, True, False, False, False, False],
        "target_size": 0.4,
    }, index=index)
    strategy = GeneratedStrategy()
    params = CapacityParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result.loc[index[2], "size"] == 0.4
    data.loc[index[2], "close"] = 96.0
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    data.loc[index[2], ["open", "close"]] = [110.0, 106.0]
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    actual = strategy.indicators(data, params)
    assert not actual["confirmed"].any()
    assert np.isfinite(actual["target_size"]).all()


if __name__ == "__main__":
    _self_check()
