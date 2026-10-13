from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    atr_period: int = 14
    strength_window: int = 3
    min_streak: int = 2
    min_strength: float = 0.9
    min_gap_drag: float = 0.1
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791333164"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.atr_period, params.strength_window, params.min_streak) + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.atr_period, params.strength_window, params.min_streak, params.hold_bars) < 1:
            raise ValueError("Periods and streak length must be positive")
        if not all(np.isfinite(v) and v > 0 for v in (params.min_strength, params.min_gap_drag, params.stop_atr)):
            raise ValueError("Strength and stop parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["body"] = data["close"] - data["open"]
        out["overnight_gap"] = data["open"] - previous_close
        up_body = out["body"].gt(0)
        out["buying_streak"] = up_body.astype(int).groupby((~up_body).cumsum()).cumsum().clip(upper=params.min_streak)
        denominator = out["atr"].where(out["atr"].gt(0))
        out["daytime_strength"] = out["body"].rolling(params.strength_window, min_periods=params.strength_window).sum() / denominator
        out["overnight_drag"] = -out["overnight_gap"].rolling(params.strength_window, min_periods=params.strength_window).sum() / denominator
        out["setup"] = (
            out["buying_streak"].ge(params.min_streak)
            & out["daytime_strength"].ge(params.min_strength)
            & out["overnight_drag"].ge(params.min_gap_drag)
            & np.isfinite(out["daytime_strength"])
            & np.isfinite(out["overnight_drag"])
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        entry_atr = 0.0
        fixed_stop = 0.0
        held = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                    held = 0
                    active = True
                else:
                    continue
            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or held >= params.hold_bars:
                    active = False
                    continue
                decisions[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_atr = atr[i]
                pending = True
                decisions[i] = 1
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "confirmed": [False, False, True, False, False, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[4], "close"] = 96.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert not derived["confirmed"].any()
    assert result["size"].gt(0).all()
