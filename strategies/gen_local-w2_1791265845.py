from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.4
    frac_window: int = 64
    baseline_window: int = 63
    discount_z: float = 0.5
    atr_period: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791265845"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.frac_window + params.baseline_window, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (
            0 < params.frac_order < 1
            and params.frac_window >= 2
            and params.baseline_window >= 2
            and params.atr_period >= 1
            and params.discount_z > 0
            and params.stop_atr > 0
            and 15 <= params.hold_bars <= 20
        ):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        fractional = close.copy()
        weight = 1.0
        # Fixed-width fractional differencing keeps the lookback bounded.
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        reference = fractional.shift(1).rolling(params.baseline_window)
        mean = reference.mean()
        std = reference.std(ddof=0).replace(0.0, np.nan)
        z = ((fractional - mean) / std).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.replace([np.inf, -np.inf], np.nan).rolling(params.atr_period).mean()
        winter = pd.Series(data.index.month.isin([11, 12, 1]), index=data.index)
        change = fractional.diff()
        setup = winter & z.lt(-params.discount_z) & change.gt(0) & atr.gt(0)
        return pd.DataFrame({
            "fractional_close": fractional,
            "reference_mean": mean,
            "reference_std": std,
            "discount_z": z,
            "fractional_change": change,
            "true_range": tr,
            "atr": atr,
            "winter": winter,
            "setup": setup.fillna(False),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        setup = indicators["setup"].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        stop = np.nan
        entry_bar = -1
        for i in range(len(data)):
            # Capture the next opening price only after the entry decision.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                stopped = np.isfinite(closes[i]) and closes[i] < stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    active = False
                else:
                    signals[i] = 1
            elif (
                confirmed[i]
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(closes[i]) and closes[i] > 0
            ):
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.bdate_range("2020-11-02", periods=30)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"setup": False, "atr": 1.0}, index=index)
    indicators.iloc[:2, 0] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20:].eq(0).all()
    data.loc[index[2], ["open", "close"]] = [110.0, 108.0]
    data.loc[index[3], "close"] = 107.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[2:4].eq(1).all()
    assert result["signal"].iloc[4:].eq(0).all()
    assert result["size"].gt(0).all()
    assert strategy.indicators(data.iloc[:10], params)["setup"].eq(False).all()


if __name__ == "__main__":
    _self_check()
