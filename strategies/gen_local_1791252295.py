from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    ma_window: int = 20
    gap_window: int = 3
    atr_window: int = 14
    entry_z: float = 1.2
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1791252295"

    @classmethod
    def params_type(cls):
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return max(200, params.ma_window, params.gap_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].where(out["std"] > 0)
        out["regime_ma"] = close.rolling(200).mean()
        out["gap"] = np.log(opening / previous)
        out["intraday"] = np.log(close / opening)
        out["gap_sum"] = out["gap"].rolling(params.gap_window).sum()
        out["intraday_sum"] = out["intraday"].rolling(params.gap_window).sum()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapParams
    ) -> SignalFrame:
        eligible = (
            (data["close"] > indicators["regime_ma"])
            & (indicators["z"] <= -params.entry_z)
            & (indicators["gap"] < 0)
            & (indicators["gap_sum"] < 0)
            & (indicators["intraday_sum"] > 0)
            & (indicators["gap_sum"] + indicators["intraday_sum"] < 0)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The prior close's decision fills at this bar's open.
                    if not np.isfinite(opening[i]) or opening[i] <= 0:
                        entry_bar = -1
                        continue
                    fixed_stop = opening[i] - params.stop_atr * atr[i - 1]
                if (not np.isfinite(close[i]) or close[i] < fixed_stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif eligible[i]:
                signal[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame({
        "open": [100., 110., 108., 106.] + [100.] * 7,
        "close": [100., 110., 108., 106.] + [100.] * 7,
    }, index=index)
    ind = pd.DataFrame({
        "regime_ma": 90., "z": np.nan, "gap": -0.02,
        "gap_sum": -0.03, "intraday_sum": 0.01, "atr": 100.,
    }, index=index)
    ind.loc[index[[0, 5]], "z"] = -2.
    ind.loc[index[[0, 5]], "atr"] = 2.
    result = GeneratedStrategy.generate_signals(None, data, ind, None, GapParams())
    # Entry-open stop stays at 107 despite the later ATR jumping to 100.
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
