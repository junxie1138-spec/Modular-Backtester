from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_length: int = 4
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791238608"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(15, params.streak_length + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.streak_length < 1 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("streak_length must be positive and stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["daily_range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["daily_range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["support_step"] = (
            data["low"].gt(out["previous_low"])
            & data["close"].gt(out["midpoint"])
            & data[["open", "high", "low", "close"]].notna().all(axis=1)
        )
        out["streak"] = out["support_step"].astype(int).groupby((~out["support_step"]).cumsum()).cumsum()
        out["fortnight_change"] = data["close"].diff(10)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        eligible = (
            indicators["streak"].ge(params.streak_length)
            & indicators["fortnight_change"].gt(0)
            & indicators["atr"].gt(0)
            & np.isfinite(indicators["atr"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_bar = 0
        fixed_stop = 0.0
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or i - entry_bar >= 19:
                    active = False
                else:
                    signals[i] = 1
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=65, freq="B")
    close = 100.0 + np.arange(65) * 0.25
    data = pd.DataFrame({"open": close - 0.1, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = StreakParams()
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert strategy.warmup_bars(params) >= 15
    assert result["signal"].iloc[:14].eq(0).all()
    assert result["signal"].iloc[14:34].eq(1).all()
    assert result["signal"].iloc[34] == 0
    shocked = data.copy()
    shocked.loc[index[18], ["high", "low", "close"]] = [80.5, 79.5, 80.0]
    stopped = strategy.generate_signals(shocked, strategy.indicators(shocked, params), None, params).data
    assert stopped["signal"].iloc[18] == 1
    assert stopped["signal"].iloc[19] == 0
    assert stopped["signal"].isin([0, 1]).all()
    assert stopped["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
