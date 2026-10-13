from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    correlation_window: int = 20
    echo_threshold: float = 0.20
    atr_window: int = 14
    stop_atr: float = 1.50
    holding_bars: int = 4
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791264095"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(params.correlation_window + 4, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if params.correlation_window < 3 or params.atr_window < 1:
            raise ValueError("Invalid indicator windows")
        if not 0.0 <= params.echo_threshold <= 1.0:
            raise ValueError("echo_threshold must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive and finite")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        if not np.isfinite(params.entry_slippage_bps) or params.entry_slippage_bps < 0.0:
            raise ValueError("entry_slippage_bps must be nonnegative and finite")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["return"] = data["close"].div(previous_close).sub(1.0).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        window = params.correlation_window
        out["correlation_lag1"] = out["return"].rolling(window, min_periods=window).corr(out["return_lag1"]).replace([np.inf, -np.inf], np.nan)
        out["correlation_lag2"] = out["return"].rolling(window, min_periods=window).corr(out["return_lag2"]).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["setup"] = (
            out["return"].gt(0.0)
            & out["correlation_lag2"].ge(params.echo_threshold)
            & out["correlation_lag1"].le(0.0)
            & out["atr"].gt(0.0)
            & np.isfinite(out["atr"])
        )
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        targets = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    # The prior close's decision fills at this open; never inspect a future open.
                    entry_price = opens[i] * (1.0 + params.entry_slippage_bps / 10000.0)
                    fixed_stop = entry_price - params.stop_atr * atr[i - 1]
                    entry_bar = i
                    active = True
            if active:
                stop_hit = np.isfinite(closes[i]) and closes[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.holding_bars
                if stop_hit or expired:
                    active = False
                else:
                    targets[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                pending = True
                targets[i] = 1
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 97.0, "close": [100.0, 100.0, 103.0, 100.0, 98.0, 98.0, 98.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "confirmed": [True, False, False, False, False, False, False]}, index=index)
    params = EchoParams(stop_atr=1.0, holding_bars=5, entry_slippage_bps=0.0)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    params.holding_bars = 3
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
