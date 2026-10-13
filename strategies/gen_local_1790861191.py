from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    stretch: float = 1.0
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790861191"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 42

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.stretch) or params.stretch <= 0:
            raise ValueError("stretch must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        close = data["close"]
        mean = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std(ddof=0)
        z = ((close - mean) / std.where(std > 0)).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["ma"] = mean
        out["std"] = std
        out["z"] = z
        out["rho1"] = z.rolling(20, min_periods=20).corr(z.shift(1)).clip(-1, 1)
        out["rho2"] = z.rolling(20, min_periods=20).corr(z.shift(2)).clip(-1, 1)
        out["tr"] = tr
        out["atr"] = tr.rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        setup = (
            (indicators["z"] < -params.stretch)
            & (indicators["rho1"] < -0.15)
            & (indicators["rho2"] > 0.15)
        ).fillna(False).to_numpy(dtype=bool)
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        age = 0
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # Entry occurs at this open; freeze the prior decision bar's ATR.
                stop = opening[i] - params.stop_atr * atr[i - 1]
                pending = False
                age = 0
            if active:
                age += 1
                if (
                    not np.isfinite(close[i])
                    or not np.isfinite(stop)
                    or close[i] < stop
                    or age >= 2
                ):
                    active = False
                else:
                    raw[i] = 1
            elif setup[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[4], "close"] = 97.0
    ind = pd.DataFrame({"z": 0.0, "rho1": -0.5, "rho2": 0.5,
                        "atr": 1.0}, index=index)
    ind.loc[index[[3, 8]], "z"] = -2.0
    result = GeneratedStrategy.generate_signals(data, ind, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
