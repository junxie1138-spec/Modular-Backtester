from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    return_window: int = 15
    range_window: int = 5
    atr_window: int = 14
    efficiency_min: float = 0.35
    range_flow_min: float = 0.10
    stop_atr: float = 2.5
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.return_window, self.range_window, self.atr_window, self.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not all(np.isfinite(x) for x in (self.efficiency_min, self.range_flow_min, self.stop_atr)):
            raise ValueError("Thresholds must be finite.")
        if not (0 < self.efficiency_min <= 1 and 0 < self.range_flow_min <= 1 and self.stop_atr > 0):
            raise ValueError("Invalid entry thresholds or stop multiplier.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791216476"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.return_window, params.range_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["returns"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["net_return"] = out["returns"].rolling(params.return_window, min_periods=params.return_window).sum()
        out["gross_return"] = out["returns"].abs().rolling(params.return_window, min_periods=params.return_window).sum()
        out["efficiency"] = out["net_return"] / out["gross_return"].replace(0, np.nan)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["upper_extension"] = data["high"].diff().clip(lower=0)
        out["lower_extension"] = (-data["low"].diff()).clip(lower=0)
        out["range_replacement"] = (out["upper_extension"] - out["lower_extension"]) / out["true_range"].replace(0, np.nan)
        out["range_flow"] = out["range_replacement"].rolling(params.range_window, min_periods=params.range_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = (
            indicators["efficiency"].ge(params.efficiency_min)
            & indicators["range_flow"].ge(params.range_flow_min)
            & indicators["atr"].gt(0)
            & np.isfinite(indicators["atr"])
            & np.isfinite(data["close"])
            & data["close"].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        pending_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            # An entry decided yesterday fills at today's open.
            if np.isfinite(pending_atr):
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * pending_atr
                pending_atr = np.nan
            if entry_bar >= 0:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif ready[i]:
                signal[i] = 1
                pending_atr = atr[i]
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 100., 110., 120., 109., 107.],
        "high": [101., 101., 121., 121., 110., 108.],
        "low": [99., 99., 109., 108., 106., 106.],
        "close": [100., 100., 120., 109., 107., 107.],
        "volume": [1000.] * 6,
    }, index=index)
    features = pd.DataFrame({
        "efficiency": [0., 1., 0., 0., 0., 0.],
        "range_flow": [0., 1., 0., 0., 0., 0.],
        "atr": [1., 1., 10., 10., 10., 10.],
    }, index=index)
    result = strategy.generate_signals(data, features, None, Params(stop_atr=2.))
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    timed = strategy.generate_signals(data, features, None, Params(stop_atr=2., hold_bars=2))
    assert timed.data["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()
    assert strategy.warmup_bars(Params()) == 16
    derived = strategy.indicators(data, Params())
    assert derived.index.equals(index)
    assert not np.isinf(derived.to_numpy(dtype=float)).any()


if __name__ == "__main__":
    _self_check()
