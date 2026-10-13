from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    roc_window: int = 5
    echo_window: int = 40
    atr_window: int = 14
    min_corr: float = 0.20
    min_echo: float = 0.25
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1791274342"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return max(params.roc_window + params.echo_window + 3,
                   params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if min(params.roc_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if params.echo_window < 3 or params.stop_atr <= 0:
            raise ValueError("Echo window must be at least three; stop must be positive")
        if not 0 < params.min_corr < 1 or params.min_echo <= 0:
            raise ValueError("Correlation and echo thresholds must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(params.roc_window, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["lag1"] = out["acceleration"].shift(1)
        out["lag2"] = out["acceleration"].shift(2)
        out["corr1"] = out["acceleration"].rolling(params.echo_window).corr(out["lag1"])
        out["corr2"] = out["acceleration"].rolling(params.echo_window).corr(out["lag2"])
        out["noise"] = out["acceleration"].rolling(params.echo_window).std(ddof=0)
        out["echo_score"] = out["corr2"] * out["lag1"] / out["noise"].where(out["noise"] > 0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        eligible = (
            (indicators["roc"] > 0)
            & (indicators["acceleration"] < 0)
            & (indicators["lag1"] > 0)
            & (indicators["corr1"] < -params.min_corr)
            & (indicators["corr2"] > params.min_corr)
            & (indicators["echo_score"] > params.min_echo)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        strength = (0.30 + 0.65 * (indicators["echo_score"] / 2.0).clip(0, 1)).fillna(0.30).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decision = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.30, dtype=float)
        active = False
        entry_bar = 0
        stop = 0.0
        entry_size = 0.30
        for i in range(len(data)):
            if not active and i > 0 and decision[i - 1] == 1:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                active = True
                entry_bar = i
                # Freeze the stop at the next-bar open using decision-bar ATR.
                stop = opens[i] - params.stop_atr * atr[i - 1]
                entry_size = sizes[i - 1]
            if active:
                sizes[i] = entry_size
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                else:
                    decision[i] = 1
            elif eligible[i]:
                decision[i] = 1
                sizes[i] = strength[i]
        df = pd.DataFrame({"signal": decision, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.30).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8, "high": [110.0] * 8,
        "low": [90.0] * 8,
        "close": [100.0, 110.0, 100.0, 94.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    ind = pd.DataFrame({
        "roc": [0.01] * 8, "acceleration": [-0.01] * 8,
        "lag1": [0.01] * 8, "corr1": [-0.5] * 8,
        "corr2": [0.5] * 8, "echo_score": [0.0] * 8,
        "atr": [2.0] * 8,
    }, index=index)
    ind.iloc[0, ind.columns.get_loc("echo_score")] = 1.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, EchoParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.625)
    assert (result["size"] > 0).all()
    ind["atr"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, EchoParams(hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    ind["echo_score"] = np.nan
    assert not GeneratedStrategy.generate_signals(None, data, ind, None, EchoParams()).data["signal"].any()


if __name__ == "__main__":
    _self_check()
