from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringParams:
    compression: float = 0.55
    stop_atr: float = 1.8


class GeneratedStrategy(BaseStrategy[SpringParams]):
    strategy_id = "gen_local_1791319188"

    @classmethod
    def params_type(cls) -> type[SpringParams]:
        return SpringParams

    def warmup_bars(self, params: SpringParams) -> int:
        return 68

    def indicators(self, data: pd.DataFrame, params: SpringParams) -> pd.DataFrame:
        if not 0.0 < params.compression < 1.0:
            raise ValueError("compression must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        for half in (1, 2, 4, 8):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
        out["fast_energy"] = (out["haar_1"].pow(2) + out["haar_2"].pow(2)).rolling(4, min_periods=4).mean()
        out["slow_energy"] = (out["haar_4"].pow(2) + out["haar_8"].pow(2)).rolling(4, min_periods=4).mean()
        out["energy_baseline"] = out["fast_energy"].rolling(48, min_periods=48).median().shift(1)
        out["energy_ratio"] = out["fast_energy"] / out["slow_energy"].replace(0.0, np.nan)
        out["compressed"] = (
            (out["fast_energy"] < params.compression * out["energy_baseline"])
            & (out["energy_ratio"] < 0.35)
        ).fillna(False)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["release"] = (
            out["compressed"].shift(1, fill_value=False)
            & (out["fast_energy"] > out["fast_energy"].shift(1))
            & (out["haar_1"].abs() > out["haar_1"].shift(1).abs())
            & (out["haar_1"] * out["haar_8"] < 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        out["candidate"] = np.select([
            out["release"] & (out["haar_1"] > 0.0),
            out["release"] & (out["haar_1"] < 0.0),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SpringParams) -> SignalFrame:
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be finite and positive")
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        for i in range(len(data)):
            if side:
                # The entry order fills at this bar's open; its ATR was fixed at the prior close.
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        side = 0
                    else:
                        stop = opens[i] - side * params.stop_atr * entry_atr
                if side:
                    stopped = (closes[i] < stop) if side == 1 else (closes[i] > stop)
                    if not np.isfinite(closes[i]) or stopped or i - entry_bar + 1 >= 4:
                        side = 0
            elif (candidates[i] in (-1, 1) and np.isfinite(closes[i])
                  and closes[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0):
                side = int(candidates[i])
                entry_bar = i + 1
                entry_atr = float(atr[i])
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SpringParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "candidate": [1, 0, 0, 0, -1, 0, 0, 0]}, index=index)
    data.loc[index[1], "open"] = 110.0
    data.loc[index[1], "close"] = 107.0
    data.loc[index[5], "close"] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, -1, 0, 0]
    data.loc[:, "open"] = 100.0
    data.loc[:, "close"] = 100.0
    indicators.loc[:, "candidate"] = 0
    indicators.loc[index[0], "candidate"] = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=100))
    derived = strategy.indicators(flat, params)
    assert derived.index.equals(flat.index) and derived["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
