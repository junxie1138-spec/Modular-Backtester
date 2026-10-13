from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InterferenceParams:
    volatility_window: int = 20
    atr_window: int = 14
    regime_width: float = 0.15
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[InterferenceParams]):
    strategy_id = "gen_local_1791229836"

    @classmethod
    def params_type(cls) -> type[InterferenceParams]:
        return InterferenceParams

    def warmup_bars(self, params: InterferenceParams) -> int:
        if params.volatility_window < 3 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive; volatility_window must be at least three.")
        if not 0.0 < params.regime_width < 1.0:
            raise ValueError("regime_width must lie between zero and one.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive and finite.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        return int(max(params.volatility_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: InterferenceParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["log_return"] = np.log(close).diff()
        out["two_day_return"] = out["log_return"].rolling(2, min_periods=2).sum()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["daily_volatility"] = out["log_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["two_day_volatility"] = out["two_day_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        denominator = np.sqrt(2.0) * out["daily_volatility"].where(
            out["daily_volatility"] > 1e-12
        )
        out["interference"] = (out["two_day_volatility"] / denominator).replace(
            [np.inf, -np.inf], np.nan
        )
        valid = (
            np.isfinite(out["interference"])
            & np.isfinite(out["atr"])
            & (out["atr"] > 0.0)
            & np.isfinite(out["log_return"])
        )
        out["regime"] = np.select([
            valid & (out["interference"] > 1.0 + params.regime_width),
            valid & (out["interference"] < 1.0 - params.regime_width),
        ], [1, -1], default=0).astype(int)
        out["candidate"] = (
            np.sign(out["log_return"].fillna(0.0)) * out["regime"]
        ).astype(int)
        confirmed = (
            out["candidate"].ne(0)
            & out["candidate"].eq(out["candidate"].shift(1))
            & out["regime"].eq(out["regime"].shift(1))
        )
        out["confirmed"] = out["candidate"].where(confirmed, 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InterferenceParams,
    ) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        entry_atr = 0.0
        fixed_stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        raise ValueError("Entry open must be positive and finite.")
                    # Anchor only when the next bar arrives; freeze the decision-bar ATR.
                    fixed_stop = opens[i] - position * params.stop_atr * entry_atr
                if not np.isfinite(closes[i]) or closes[i] <= 0.0:
                    raise ValueError("An active position requires a positive finite close.")
                stopped = (
                    closes[i] < fixed_stop if position == 1
                    else closes[i] > fixed_stop
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
            elif (
                entries[i] != 0
                and np.isfinite(atr[i])
                and atr[i] > 0.0
                and np.isfinite(closes[i])
                and closes[i] > 0.0
            ):
                position = int(entries[i])
                entry_bar = i + 1
                entry_atr = atr[i]
                fixed_stop = np.nan
            desired[i] = position
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = InterferenceParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 110.0, "low": 90.0,
        "close": [100.0, 105.0, 98.0, 96.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"atr": 2.0, "confirmed": [1, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    # A rise to 105 must not raise the fixed 97 stop.
    data["close"] = [100.0, 95.0, 102.0, 104.0, 100.0, 100.0]
    ind["confirmed"] = [-1, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0]
    data["close"] = 100.0
    ind["confirmed"] = [1, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    trend_index = pd.date_range("2020-01-01", periods=50)
    prices = 100.0 * np.exp(np.cumsum(np.linspace(0.001, 0.03, 50)))
    trend = pd.DataFrame({
        "open": prices, "high": prices + 1.0, "low": prices - 1.0,
        "close": prices, "volume": 1000.0,
    }, index=trend_index)
    computed = strategy.indicators(trend, params)
    first = np.flatnonzero(computed["candidate"].to_numpy())[0]
    assert computed["confirmed"].iloc[first] == 0
    assert computed["confirmed"].iloc[first + 1] == 1
    assert strategy.warmup_bars(params) >= first + 2


if __name__ == "__main__":
    _self_check()
