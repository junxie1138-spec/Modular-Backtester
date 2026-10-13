from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    regime_window: int = 20
    overnight_share: float = 0.45
    streak_length: int = 2
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791291655"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.regime_window + 2, params.atr_window + 1, params.streak_length + 2))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.regime_window < 2 or params.atr_window < 2 or params.streak_length < 1:
            raise ValueError("Lookbacks must be positive; regime and ATR windows must be at least two.")
        if not 0 < params.overnight_share < 1 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Invalid regime threshold or stop multiplier.")
        if not 3 <= params.hold_days <= 5:
            raise ValueError("hold_days must be between three and five.")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["gap_return"] = (data["open"] / previous.where(previous > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["session_return"] = (data["close"] / data["open"].where(data["open"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["gap_energy"] = out["gap_return"].pow(2).rolling(params.regime_window).sum()
        out["session_energy"] = out["session_return"].pow(2).rolling(params.regime_window).sum()
        total_energy = out["gap_energy"] + out["session_energy"]
        out["overnight_share"] = out["gap_energy"] / total_energy.where(total_energy > 0)
        out["shock_regime"] = out["overnight_share"].ge(params.overnight_share)
        valid = data[["open", "high", "low", "close"]].gt(0).all(axis=1) & np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        out["repair_event"] = valid & out["gap_return"].lt(0) & data["close"].gt(previous)
        out["accept_event"] = valid & out["gap_return"].gt(0) & out["session_return"].gt(0)
        for event, streak in (("repair_event", "repair_streak"), ("accept_event", "accept_streak")):
            mask = out[event]
            out[streak] = mask.astype(int).groupby((~mask).cumsum()).cumsum().clip(upper=params.streak_length)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(valid & previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ready = out["overnight_share"].notna() & np.isfinite(out["atr"]) & out["atr"].gt(0)
        out["candidate"] = ready & (
            (out["shock_regime"] & out["repair_streak"].ge(params.streak_length))
            | (~out["shock_regime"] & out["accept_streak"].ge(params.streak_length))
        )
        out["confirmed"] = (
            out["candidate"]
            & out["candidate"].shift(1, fill_value=False)
            & out["shock_regime"].eq(out["shock_regime"].shift(1))
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Anchor to the next-bar opening fill; freeze decision-bar ATR.
                    stop = opening[i] - params.stop_atr * atr[i - 1]
                if not np.isfinite(stop) or not np.isfinite(close[i]) or close[i] < stop or i - entry_bar + 1 >= params.hold_days:
                    active = False
                else:
                    signal[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]):
                active = True
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams(regime_window=2, atr_window=2)
    strategy = GeneratedStrategy()
    opening = 100 * 1.011 ** np.arange(12)
    data = pd.DataFrame({"open": opening, "high": opening * 1.01 + 0.2, "low": opening - 0.2, "close": opening * 1.01, "volume": 100.0}, index=pd.date_range("2020-01-01", periods=12))
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    assert np.flatnonzero(indicators["confirmed"].to_numpy())[0] == 3
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].iloc[:4].eq(0).all()
    assert signals["signal"].iloc[4:9].tolist() == [1, 1, 1, 1, 0]
    probe = pd.DataFrame({"open": 100.0, "high": 111.0, "low": 99.0, "close": 100.0, "volume": 100.0}, index=data.index[:8])
    probe.loc[probe.index[2:4], "close"] = [110.0, 105.0]
    manual = pd.DataFrame({"atr": 1.0, "confirmed": False}, index=probe.index)
    manual.loc[manual.index[1], "confirmed"] = True
    fixed = strategy.generate_signals(probe, manual, ctx, params).data
    assert fixed["signal"].iloc[2:7].tolist() == [1, 1, 1, 1, 0]
    probe.loc[probe.index[2], ["open", "close"]] = [110.0, 107.0]
    stopped = strategy.generate_signals(probe, manual, ctx, params).data
    assert stopped["signal"].iloc[2:4].tolist() == [1, 0]
    empty = strategy.generate_signals(data.iloc[:0], strategy.indicators(data.iloc[:0], params), ctx, params).data
    assert empty.empty


if __name__ == "__main__":
    _self_check()
