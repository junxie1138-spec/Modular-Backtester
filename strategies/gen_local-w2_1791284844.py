from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_ratio: float = 1.25
    atr_multiple: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791284844"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.volume_ratio) or params.volume_ratio <= 0:
            raise ValueError("volume_ratio must be positive and finite")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive and finite")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["prior_volume"] = data["volume"].shift(1).rolling(20, min_periods=20).mean()
        out["volume_ratio"] = data["volume"] / out["prior_volume"].where(out["prior_volume"] > 0)
        out["return"] = data["close"] / previous.where(previous > 0) - 1.0
        spread = data["high"] - data["low"]
        out["close_location"] = (data["close"] - data["low"]) / spread.where(spread > 0)
        weekday = data.index.dayofweek
        friday_day = data.index.day + (4 - weekday) % 7
        out["expiry_week"] = (weekday < 4) & (friday_day >= 15) & (friday_day <= 21)
        out["entry"] = (
            out["expiry_week"]
            & (out["volume_ratio"] >= params.volume_ratio)
            & (out["return"] > 0)
            & (data["close"] > data["open"])
            & (out["close_location"] >= 0.75)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        desired = np.zeros(len(data), dtype=int)
        active = False
        entry = peak = stop = 0.0
        armed = False
        age = 0
        last_month = -1
        for i in range(len(data)):
            close = float(data["close"].iat[i])
            atr = float(indicators["atr"].iat[i])
            if active:
                if age == 0:
                    # Entry uses the next bar's open, after yesterday's decision.
                    entry = float(data["open"].iat[i])
                    prior_atr = float(indicators["atr"].iat[i - 1])
                    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(prior_atr) or prior_atr <= 0:
                        active = False
                        continue
                    peak = entry
                    stop = max(0.0, entry - params.atr_multiple * prior_atr)
                age += 1
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    active = False
                    continue
                # Stops are evaluated at the close; exits fill next session.
                if close <= stop or age >= 18:
                    active = False
                    continue
                peak = max(peak, close)
                armed = armed or close >= entry * 1.02
                if armed:
                    stop = max(stop, entry, peak - params.atr_multiple * atr)
                if close <= stop:
                    active = False
                    continue
                desired[i] = 1
            else:
                month = data.index[i].year * 12 + data.index[i].month
                if i >= self.warmup_bars(params) - 1 and bool(indicators["entry"].iat[i]) and month != last_month:
                    active = True
                    armed = False
                    age = 0
                    last_month = month
                    desired[i] = 1
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    idx = pd.bdate_range("2022-01-03", periods=50)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 98.0,
        "close": 100.0, "volume": 1000.0,
    }, index=idx)
    data.iloc[21, data.columns.get_loc("close")] = 103.0
    data.iloc[22, data.columns.get_loc("close")] = 99.0
    strategy = GeneratedStrategy()
    params = Params()
    ind = strategy.indicators(data, params)
    assert ind.index.equals(data.index)
    assert not ind["entry"].iloc[:20].any()
    ind["atr"] = 1.0
    ind["entry"] = False
    ind.iloc[[20, 25], ind.columns.get_loc("entry")] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=21)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[20] == 0 and result["signal"].iloc[21] == 1
    assert result["signal"].iloc[22] == 1 and result["signal"].iloc[23] == 0
    assert result["signal"].iloc[26] == 1
    assert result["signal"].iloc[43] == 1 and result["signal"].iloc[44] == 0
    assert set(result["signal"].unique()) <= {0, 1}
    assert (result["size"] > 0).all()
    prefix = strategy.generate_signals(data.iloc[:30], ind.iloc[:30], ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:30], prefix)


if __name__ == "__main__":
    _self_check()
