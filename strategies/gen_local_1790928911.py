from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_bars: int = 40
    vote_bars: int = 6
    dominance: float = 0.45
    atr_bars: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790928911"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.range_bars + params.vote_bars + 1, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.range_bars, params.vote_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not 0 < params.dominance < 1 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Invalid dominance or stop multiplier.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["prior_high"] = data["high"].rolling(params.range_bars).max().shift(1)
        out["prior_low"] = data["low"].rolling(params.range_bars).min().shift(1)
        out["range_width"] = (out["prior_high"] - out["prior_low"]).replace(0, np.nan)
        out["position"] = (close - out["prior_low"]) / out["range_width"]
        volume = data["volume"].where(data["volume"] >= 0)
        valid = out["position"].notna()
        out["volume_sum"] = volume.where(valid).rolling(params.vote_bars).sum()
        out["upper_volume"] = volume.where(out["position"] >= 2 / 3, 0.0).where(valid).rolling(params.vote_bars).sum()
        out["lower_volume"] = volume.where(out["position"] <= 1 / 3, 0.0).where(valid).rolling(params.vote_bars).sum()
        out["dominance"] = (out["upper_volume"] - out["lower_volume"]) / out["volume_sum"].replace(0, np.nan)
        out["prior_dominance"] = out["dominance"].shift(1)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["ma200", "position", "dominance", "prior_dominance", "atr"]].notna().all(axis=1) & (out["atr"] > 0)
        long_entry = ready & (close > out["ma200"]) & out["position"].between(2 / 3, 0.95) & (out["dominance"] > params.dominance) & (out["prior_dominance"] <= params.dominance)
        short_entry = ready & (close < out["ma200"]) & out["position"].between(0.05, 1 / 3) & (out["dominance"] < -params.dominance) & (out["prior_dominance"] >= -params.dominance)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        side = 0
        pending = False
        stop = np.nan
        age = 0
        for i in range(len(data)):
            if side:
                if pending:
                    # Today's open is observed only today; ATR is frozen at yesterday's decision.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        pending = False
                        continue
                    stop = opens[i] - side * params.stop_atr * atrs[i - 1]
                    age = 0
                    pending = False
                age += 1
                stopped = np.isfinite(closes[i]) and side * (closes[i] - stop) < 0
                if stopped or age >= params.hold_bars:
                    side = 0
                signals[i] = side
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                side = int(entries[i])
                pending = True
                signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params(stop_atr=2.5, hold_bars=18)
    index = pd.date_range("2020-01-01", periods=22)
    for side in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 2.0, "entry": 0}, index=index)
        ind.iloc[0, ind.columns.get_loc("entry")] = side
        data.iloc[1, data.columns.get_loc("close")] = 100.0 + side * 10.0
        data.iloc[3, data.columns.get_loc("close")] = 100.0 - side * 6.0
        signal = GeneratedStrategy.generate_signals(None, data, ind, None, params).data["signal"]
        assert signal.iloc[:5].tolist() == [0, side, side, side, 0]
        data["close"] = 100.0
        signal = GeneratedStrategy.generate_signals(None, data, ind, None, params).data["signal"]
        assert (signal.iloc[1:19] == side).all() and (signal.iloc[19:] == 0).all()
    strategy = GeneratedStrategy()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0}, index=pd.date_range("2020-01-01", periods=250))
    ind = strategy.indicators(flat, params)
    assert (ind["entry"] == 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
