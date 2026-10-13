from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    momentum_window: int = 12
    participation_threshold: float = 0.45
    atr_window: int = 14
    trail_atr: float = 0.9
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791208307"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.momentum_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.momentum_window < 2 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; momentum_window must be at least two.")
        if not 0.0 < params.participation_threshold < 1.0:
            raise ValueError("participation_threshold must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["up_return"] = out["return"].clip(lower=0.0)
        out["up_return_squared"] = out["up_return"].pow(2)
        out["up_sum"] = out["up_return"].rolling(params.momentum_window).sum()
        out["up_square_sum"] = out["up_return_squared"].rolling(params.momentum_window).sum()
        out["participation"] = out["up_sum"].pow(2).div(
            params.momentum_window * out["up_square_sum"].replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        out["prior_participation"] = out["participation"].shift(1)
        out["net_return"] = out["return"].rolling(params.momentum_window).sum()
        out["ma200"] = close.rolling(200).mean()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (close > out["ma200"])
            & (out["return"] > 0.0)
            & (out["net_return"] > 0.0)
            & (out["participation"] >= params.participation_threshold)
            & (out["prior_participation"] < params.participation_threshold)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0 or params.hold_bars < 1:
            raise ValueError("trail_atr and hold_bars must be positive.")
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and np.isfinite(regime[i])
            if not valid or atr[i] <= 0.0 or close[i] <= regime[i]:
                active = False
                continue
            if active:
                # This is a held session: entry fills after the preceding decision bar.
                age += 1
                high_water = close[i] if age == 1 else max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 103.0, 102.0, 101.0, 104.0]}, index=index)
    ind = pd.DataFrame(
        {"entry": [True, False, False, False, False],
         "atr": [1.0, 1.0, 10.0, 10.0, 1.0], "ma200": 90.0},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(None, data, ind, None, Params(hold_bars=4))
    # Widening ATR cannot lower the stop established at the first held close.
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    data["close"] = [100.0, 101.0, 102.0, 103.0, 104.0]
    result = GeneratedStrategy.generate_signals(None, data, ind, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result.data["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
