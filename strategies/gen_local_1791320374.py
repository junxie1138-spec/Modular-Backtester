from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_lookback: int = 20
    breakout_lookback: int = 5
    opening_sessions: int = 3
    volume_multiplier: float = 1.4
    minimum_move: float = 0.004
    breakeven_gain: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791320374"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.volume_lookback + 1,
                   params.breakout_lookback + 1, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["return"] = close.div(previous_close.where(previous_close > 0)).sub(1)
        average_volume = data["volume"].shift(1).rolling(
            params.volume_lookback, min_periods=params.volume_lookback
        ).mean()
        out["volume_ratio"] = data["volume"].div(
            average_volume.where(average_volume > 0)
        )
        out["prior_high"] = data["high"].shift(1).rolling(
            params.breakout_lookback, min_periods=params.breakout_lookback
        ).max()
        out["prior_low"] = data["low"].shift(1).rolling(
            params.breakout_lookback, min_periods=params.breakout_lookback
        ).min()
        month_keys = data.index.year * 12 + data.index.month
        session = pd.Series(0, index=data.index).groupby(month_keys).cumcount() + 1
        out["opening_window"] = session.le(params.opening_sessions)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if (params.volume_lookback < 1 or params.breakout_lookback < 1
                or params.opening_sessions < 1 or params.hold_bars < 1
                or not np.isfinite(params.volume_multiplier)
                or params.volume_multiplier <= 0
                or not np.isfinite(params.minimum_move)
                or params.minimum_move < 0
                or not np.isfinite(params.breakeven_gain)
                or params.breakeven_gain <= 0
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")

        ready = pd.Series(np.isfinite(indicators[[
            "atr", "ma200", "return", "volume_ratio", "prior_high", "prior_low"
        ]]).all(axis=1), index=data.index)
        ready &= indicators["atr"].gt(0)
        ready &= np.isfinite(data["close"]) & data["close"].gt(0)
        eligible = (ready & indicators["opening_window"].fillna(False)
                    & indicators["volume_ratio"].ge(params.volume_multiplier))
        long_entry = (eligible & data["close"].gt(indicators["ma200"])
                      & data["close"].gt(indicators["prior_high"])
                      & indicators["return"].ge(params.minimum_move))
        short_entry = (eligible & data["close"].lt(indicators["ma200"])
                       & data["close"].lt(indicators["prior_low"])
                       & indicators["return"].le(-params.minimum_move))
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        stop = np.nan
        armed = False
        held = 0

        for i in range(len(data)):
            if position:
                # Yesterday's decision fills at today's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        raise ValueError("Invalid entry open")
                    entry_price = opens[i]
                    stop = position * entry_price - params.trail_atr * entry_atr
                held += 1
                signed_close = position * closes[i]
                if np.isfinite(signed_close):
                    gain = position * (closes[i] / entry_price - 1.0)
                    armed = armed or gain >= params.breakeven_gain
                    if armed:
                        # Signed stops only rise; short stop prices consequently fall.
                        stop = max(stop, position * entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, signed_close - params.trail_atr * atrs[i])
                    stopped = signed_close <= stop
                else:
                    stopped = False
                if stopped or held >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    armed = False
                    held = 0
            elif entries[i]:
                position = int(entries[i])
                entry_price = np.nan
                entry_atr = atrs[i]
                armed = False
                held = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    index = pd.bdate_range("2020-01-01", periods=240)
    for side in (1, -1):
        data = pd.DataFrame({
            "open": 100.0, "high": 105.0, "low": 95.0,
            "close": 100.0, "volume": 1000.0,
        }, index=index)
        ind = pd.DataFrame({
            "atr": 1.0, "ma200": 100.0 - side * 10,
            "return": 0.0, "volume_ratio": 1.0,
            "prior_high": 101.0, "prior_low": 99.0,
            "opening_window": False,
        }, index=index)
        data.loc[index[201], "close"] = 100.0 + side * 2
        data.loc[index[202], "close"] = 100.0 + side * 4
        data.loc[index[203], "close"] = 100.0 + side
        ind.loc[index[201], ["return", "volume_ratio", "opening_window"]] = [
            side * 0.02, 2.0, True
        ]
        result = GeneratedStrategy.generate_signals(None, data, ind, None, params)
        assert result.data["signal"].iloc[201:205].tolist() == [0, side, side, 0]
        assert result.data["size"].gt(0).all()
        # Disable profit arming to exercise the eighteen-session time exit.
        data["close"] = 100.0 + side * 2
        timed = GeneratedStrategy.generate_signals(None, data, ind, None, params)
        assert timed.data["signal"].iloc[202:220].eq(side).all()
        assert timed.data["signal"].iloc[220] == 0
    actual = GeneratedStrategy.indicators(None, data, params)
    assert actual.index.equals(data.index)
    assert actual["ma200"].iloc[:199].isna().all()
    assert GeneratedStrategy.warmup_bars(None, params) >= 200


if __name__ == "__main__":
    _self_check()
