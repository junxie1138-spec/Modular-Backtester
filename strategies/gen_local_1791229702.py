from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_ratio: float = 1.3
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791229702"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 64

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not np.isfinite(params.energy_ratio) or params.energy_ratio <= 0:
            raise ValueError("energy_ratio must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        for half in (2, 4, 8):
            mean = out["return"].rolling(half, min_periods=half).mean()
            # Causal, unit-norm Haar detail over two adjacent return blocks.
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(20, min_periods=20).mean()
        fine = out["energy_2"] + out["energy_4"]
        out["energy_ratio"] = out["energy_8"] / fine.where(fine > 0)
        out["coarse_corr"] = out["haar_8"].rolling(40, min_periods=40).corr(out["haar_8"].shift(8))
        out["drift"] = out["return"].rolling(16, min_periods=16).sum()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["energy_ratio", "coarse_corr", "drift", "atr"]].notna().all(axis=1)
        setup = ready & (out["atr"] > 0) & (out["energy_ratio"] >= params.energy_ratio) & (out["coarse_corr"] > 0.2)
        out["setup"] = np.where(setup, np.sign(out["drift"].fillna(0)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].to_numpy(dtype=int)
        side = 0
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        extreme = 0.0
        age = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or min(opening, high, low, close) <= 0:
                side = 0
                pending = False
                continue
            if side == 0:
                if i >= 63 and setups[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                    side = int(setups[i])
                    pending = True
                    armed = False
                    age = 0
                    raw[i] = side
                continue
            if pending:
                # The preceding close's decision fills at this bar's open.
                entry = opening
                extreme = side * entry
                stop = side * entry - params.trail_atr * atr[i - 1]
                pending = False
            age += 1
            adverse = low if side == 1 else -high
            if adverse <= stop or age >= 18:
                side = 0
                continue
            favorable = high if side == 1 else -low
            extreme = max(extreme, favorable)
            if extreme - side * entry >= 0.02 * entry:
                armed = True
            if armed:
                # Signed stop only rises; a short's price stop only falls.
                stop = max(stop, side * entry)
                if np.isfinite(atr[i]) and atr[i] > 0:
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                # Detect a close below a newly raised stop without guessing
                # the intrabar ordering of the favorable and adverse extremes.
                if side * close <= stop:
                    side = 0
                    continue
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = WaveletParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=100)
    for side in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
        if side == 1:
            data.loc[index[71], ["high", "low", "close"]] = [104.0, 100.0, 103.0]
        else:
            data.loc[index[71], ["high", "low", "close"]] = [100.0, 96.0, 97.0]
        indicators = pd.DataFrame({"atr": 2.0, "setup": 0}, index=index)
        indicators.loc[index[70], "setup"] = side
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].iloc[70:74].tolist() == [0, side, side, 0]
        assert (result["size"] > 0).all()
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "setup": 0}, index=index)
    indicators.loc[index[70], "setup"] = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert (result["signal"].iloc[71:89] == 1).all()
    assert result["signal"].iloc[89] == 0
    assert strategy.indicators(data, params)["setup"].eq(0).all()


if __name__ == "__main__":
    _self_check()
