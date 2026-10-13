from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankRepairParams:
    fast_window: int = 8
    slow_window: int = 80
    recovery_rank: float = 0.80
    legacy_tail: float = 0.35
    min_dislocation: float = 0.02
    atr_window: int = 14
    initial_stop_atr: float = 3.0
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[RankRepairParams]):
    strategy_id = "gen_local-w2_1791210153"

    @classmethod
    def params_type(cls) -> type[RankRepairParams]:
        return RankRepairParams

    def warmup_bars(self, params: RankRepairParams) -> int:
        return int(max(200, params.slow_window, params.fast_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RankRepairParams) -> pd.DataFrame:
        if not (2 <= params.fast_window < params.slow_window
                and params.atr_window >= 1 and params.max_hold >= 1
                and 0.5 < params.recovery_rank < 1.0
                and 0.0 < params.legacy_tail < 0.5
                and 0.0 < params.min_dislocation < 1.0
                and params.initial_stop_atr > 0.0
                and params.trail_atr > 0.0 and params.breakeven_pct > 0.0):
            raise ValueError("Invalid rank-repair parameters")
        close = data["close"].where(data["close"] > 0.0)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["fast_rank"] = close.rolling(params.fast_window).rank(pct=True)
        out["slow_rank"] = close.rolling(params.slow_window).rank(pct=True)
        out["peak"] = close.rolling(params.slow_window).max()
        out["trough"] = close.rolling(params.slow_window).min()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["drawup"] = close / out["trough"] - 1.0
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        tr = tr.where(data[["high", "low"]].notna().all(axis=1) & previous.notna())
        out["atr"] = tr.rolling(params.atr_window).mean()
        out["long_setup"] = (
            (close > out["ma200"])
            & (out["drawdown"] >= params.min_dislocation)
            & (out["slow_rank"] <= params.legacy_tail)
            & (out["fast_rank"] >= params.recovery_rank)
            & (out["fast_rank"].shift(1) < params.recovery_rank)
        ).fillna(False)
        out["short_setup"] = (
            (close < out["ma200"])
            & (out["drawup"] >= params.min_dislocation)
            & (out["slow_rank"] >= 1.0 - params.legacy_tail)
            & (out["fast_rank"] <= 1.0 - params.recovery_rank)
            & (out["fast_rank"].shift(1) > 1.0 - params.recovery_rank)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RankRepairParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        regimes = indicators["ma200"].to_numpy(dtype=float)
        longs = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        side, age = 0, 0
        entry, entry_atr = np.nan, np.nan
        stop, best, armed = -np.inf, -np.inf, False
        for i in range(len(data)):
            close, atr, regime = closes[i], atrs[i], regimes[i]
            valid = (np.isfinite(close) and close > 0.0
                     and np.isfinite(atr) and atr > 0.0
                     and np.isfinite(regime))
            if side:
                # The previous close's decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0.0:
                        side, entry = 0, np.nan
                        continue
                    best = side * entry
                    stop = best - params.initial_stop_atr * entry_atr
                age += 1
                if not valid:
                    side, entry = 0, np.nan
                    continue
                # Signed-price stops only rise; short stop prices therefore fall.
                # Stops are evaluated at the close and exits fill next bar.
                best = max(best, side * close)
                if side * (close - entry) / entry >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, best - params.trail_atr * atr)
                if (side * close <= stop or age >= params.max_hold
                        or side * (close - regime) <= 0.0):
                    side, entry = 0, np.nan
            elif valid and (longs[i] or shorts[i]):
                side = 1 if longs[i] else -1
                age, entry, entry_atr = 0, np.nan, atr
                stop, best, armed = -np.inf, -np.inf, False
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RankRepairParams()
    index = pd.date_range("2020-01-01", periods=7)
    for side in (1, -1):
        closes = 100.0 + side * np.array([0.0, 3.0, 8.0, 5.0, 1.0, 0.0, 0.0])
        data = pd.DataFrame({"open": 100.0, "high": closes + 1.0,
                             "low": closes - 1.0, "close": closes,
                             "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 2.0, "ma200": 100.0 - side * 10.0,
                            "long_setup": False, "short_setup": False}, index=index)
        ind.loc[index[0], "long_setup" if side == 1 else "short_setup"] = True
        result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0, 0]
        assert (result["size"] > 0.0).all()
        for end in range(1, len(data) + 1):
            prefix = GeneratedStrategy.generate_signals(
                None, data.iloc[:end], ind.iloc[:end], None, params).data
            assert prefix.equals(result.iloc[:end])


if __name__ == "__main__":
    _self_check()
