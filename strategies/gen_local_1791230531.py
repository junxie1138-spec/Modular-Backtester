from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    history_window: int = 126
    atr_window: int = 14
    streak_quantile: float = 0.85
    drawdown_quantile: float = 0.65
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    max_hold: int = 4
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791230531"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window + params.history_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if (min(params.peak_window, params.history_window, params.atr_window) < 2
                or not 0 < params.streak_quantile < 1
                or not 0 < params.drawdown_quantile < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
                or not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0
                or not np.isfinite(params.entry_slippage_bps)
                or params.entry_slippage_bps < 0 or not 3 <= params.max_hold <= 5):
            raise ValueError("Invalid recovery parameters")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(params.peak_window).max()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].replace(0, np.nan)
        repairing = (data["low"].gt(data["low"].shift(1))
                     & out["drawdown"].gt(0)
                     & data["low"].notna() & data["close"].notna())
        out["repair_streak"] = repairing.astype(int).groupby(
            (~repairing).cumsum()).cumsum()
        out["streak_cutoff"] = out["repair_streak"].shift(1).rolling(
            params.history_window).quantile(params.streak_quantile).clip(lower=1.0)
        out["drawdown_cutoff"] = out["drawdown"].shift(1).rolling(
            params.history_window).quantile(params.drawdown_quantile)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        qualifies = (out["repair_streak"].gt(out["streak_cutoff"])
                     & out["drawdown"].gt(out["drawdown_cutoff"])
                     & out["drawdown"].gt(0) & out["atr"].gt(0))
        out["entry"] = (qualifies & ~qualifies.shift(1, fill_value=False)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        decision = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        warmup = max(self.warmup_bars(params), int(ctx.warmup_bars))
        active = False
        armed = False
        entry_price = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            open_price, high, low, close = prices[i]
            valid = (np.isfinite(prices[i]).all() and min(prices[i]) > 0
                     and np.isfinite(atr[i]) and atr[i] > 0)
            if active:
                if not valid:
                    active = False
                    continue
                if np.isnan(entry_price):
                    entry_price = open_price * (1.0 + params.entry_slippage_bps / 10000.0)
                    stop = entry_price - params.trail_atr * atr[i - 1]
                age += 1
                # Test the previously established stop before using today's high.
                if low <= stop or age >= params.max_hold:
                    active = False
                    continue
                if high >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, high - params.trail_atr * atr[i])
                    # Today's ratchet applies at the close, then to subsequent bars.
                    if close <= stop:
                        active = False
                        continue
                decision[i] = 1
            elif i >= warmup and valid and entries[i]:
                active = True
                armed = False
                entry_price = np.nan
                stop = -np.inf
                age = 0
                decision[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = decision
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RecoveryParams(peak_window=2, history_window=2, atr_window=2)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "entry": False}, index=index)
    indicators.loc[index[5], "entry"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    data.loc[index[6], ["high", "close"]] = [103.0, 102.0]
    data.loc[index[7], ["high", "low", "close"]] = [102.0, 100.0, 101.0]
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= params.peak_window + params.history_window + 1


if __name__ == "__main__":
    _self_check()
