from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    lookback: int = 20
    atr_period: int = 14
    min_drawdown: float = 0.02
    shock_multiple: float = 1.5
    migration_atr: float = 0.10
    volume_drain: float = 0.75
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791264197"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; lookback must be at least two.")
        if not (0 < params.min_drawdown < 1 and params.shock_multiple > 0
                and params.migration_atr > 0 and 0 < params.volume_drain < 1
                and params.stop_atr > 0 and params.hold_bars in (1, 2)):
            raise ValueError("Invalid recovery or risk parameters.")
        out = pd.DataFrame(index=data.index)
        out["range"] = data["high"] - data["low"]
        out["price_change"] = data["close"].diff()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["prior_peak"] = data["high"].rolling(
            params.lookback, min_periods=params.lookback
        ).max().shift(1)
        out["drawdown"] = data["close"] / out["prior_peak"].replace(0, np.nan) - 1
        out["range_base"] = out["range"].rolling(
            params.lookback, min_periods=params.lookback
        ).mean().shift(2)
        out["volume_base"] = data["volume"].rolling(
            params.lookback, min_periods=params.lookback
        ).mean().shift(2)
        out["shock_ratio"] = out["range"].shift(1) / out["range_base"].replace(0, np.nan)
        previous_atr = out["atr"].shift(1).replace(0, np.nan)
        out["high_move"] = data["high"].diff() / previous_atr
        out["low_move"] = data["low"].diff() / previous_atr
        out["volume_ratio"] = data["volume"] / data["volume"].shift(1).replace(0, np.nan)
        out["prior_volume_burst"] = data["volume"].shift(1) / out["volume_base"].replace(0, np.nan)
        shock = out["shock_ratio"].ge(params.shock_multiple)
        range_up = (shock & out["high_move"].ge(params.migration_atr)
                    & out["low_move"].ge(params.migration_atr)
                    & out["price_change"].shift(1).lt(0))
        range_down = (shock & out["high_move"].le(-params.migration_atr)
                      & out["low_move"].le(-params.migration_atr)
                      & out["price_change"].shift(1).gt(0))
        out["range_vote"] = np.select([range_up, range_down], [1, -1], default=0)
        drained = (out["volume_ratio"].le(params.volume_drain)
                   & out["prior_volume_burst"].ge(1)
                   & data["volume"].gt(0))
        out["pressure_vote"] = np.select(
            [drained & out["price_change"].gt(0),
             drained & out["price_change"].lt(0)], [1, -1], default=0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        eligible = (indicators["drawdown"].le(-params.min_drawdown)
                    & indicators["range_vote"].eq(indicators["pressure_vote"])
                    & indicators["range_vote"].ne(0)
                    & indicators["atr"].gt(0))
        candidates = indicators["range_vote"].where(eligible, 0).fillna(0).to_numpy(dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        stop = np.nan
        for i in range(len(data)):
            if position:
                if entry_bar < 0:
                    # Entry executes at this open; freeze the decision-bar ATR.
                    entry_bar = i
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                invalid = not (np.isfinite(closes[i]) and np.isfinite(stop)
                               and np.isfinite(opens[i]) and opens[i] > 0)
                stopped = position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    position, entry_bar = 0, -1
                decisions[i] = position
                continue
            if candidates[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]):
                position = int(candidates[i])
                entry_bar = -1
                decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0,
        "close": [100, 98, 100, 100, 100, 102, 100, 100, 100, 104, 101, 100],
        "volume": 1000.0,
    }, index=index)
    votes = [1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0]
    features = pd.DataFrame({
        "drawdown": -0.05, "range_vote": votes,
        "pressure_vote": votes, "atr": 1.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 1, 0]
    features["pressure_vote"] = 0
    assert strategy.generate_signals(data, features, ctx, params).data["signal"].eq(0).all()
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
