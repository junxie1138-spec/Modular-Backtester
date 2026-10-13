from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReclaimParams:
    min_gap_atr: float = 0.35
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[GapReclaimParams]):
    strategy_id = "gen_local-w2_1790978774"

    @classmethod
    def params_type(cls):
        return GapReclaimParams

    def warmup_bars(self, params: GapReclaimParams) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: GapReclaimParams) -> pd.DataFrame:
        if params.min_gap_atr <= 0 or params.trail_atr <= 0:
            raise ValueError("Parameters must be positive")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_open"] = data["open"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["up_move"] = data["high"].diff()
        out["down_move"] = -data["low"].diff()
        valid = out["up_move"].notna() & out["down_move"].notna()
        out["plus_dm"] = out["up_move"].where(
            (out["up_move"] > out["down_move"]) & (out["up_move"] > 0), 0.0
        ).where(valid)
        out["minus_dm"] = out["down_move"].where(
            (out["down_move"] > out["up_move"]) & (out["down_move"] > 0), 0.0
        ).where(valid)
        out["plus_sum"] = out["plus_dm"].rolling(14, min_periods=14).sum()
        out["minus_sum"] = out["minus_dm"].rolling(14, min_periods=14).sum()
        out["directional_strength"] = (
            (out["plus_sum"] - out["minus_sum"])
            / (out["plus_sum"] + out["minus_sum"]).replace(0.0, np.nan)
        )
        out["gap_atr"] = out["gap"] / out["prior_atr"].replace(0.0, np.nan)
        out["entry"] = (
            (out["prior_close"] < out["prior_open"])
            & (data["open"] > out["prior_open"])
            & (data["close"] > data["open"])
            & (out["gap_atr"] >= params.min_gap_atr)
            & (out["directional_strength"] >= 0.5)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReclaimParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        age = 0

        for i in range(len(data)):
            if not active:
                if entries[i]:
                    signals[i] = 1
                    active = True
                    pending_entry = True
                continue

            if pending_entry:
                entry_price = opens[i]
                stop = entry_price - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
                pending_entry = False

            age += 1
            # The existing stop applies to this bar; new ratchets use its close-time information.
            breached = lows[i] <= stop
            if highs[i] >= entry_price * 1.006:
                armed = True
            if armed:
                stop = max(stop, entry_price, highs[i] - params.trail_atr * atr[i])

            # Every exit is a close-time decision executed at the following open.
            if breached or closes[i] <= stop or age >= 2:
                active = False
            else:
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    opening = 100.0 + np.arange(40) * 0.3
    closing = opening + 0.2
    closing[35] = opening[35] - 0.1
    opening[36] = opening[35] + 0.6
    closing[36] = opening[36] + 0.2
    sample = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, closing) + 0.1,
        "low": np.minimum(opening, closing) - 0.1,
        "close": closing,
        "volume": 1000000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapReclaimParams()
    features = strategy.indicators(sample, params)
    result = strategy.generate_signals(sample, features, None, params).data
    assert features["entry"].iloc[36]
    assert result["signal"].iloc[36:40].tolist() == [0, 1, 1, 0]
    shocked = sample.copy()
    shocked.loc[index[37], "low"] = opening[37] - 5.0
    stopped = strategy.generate_signals(
        shocked, strategy.indicators(shocked, params), None, params
    ).data
    assert stopped["signal"].iloc[37:40].tolist() == [1, 0, 0]
    rallied = sample.copy()
    rallied.loc[index[37], "high"] = opening[37] * 1.02
    ratcheted = strategy.generate_signals(
        rallied, strategy.indicators(rallied, params), None, params
    ).data
    assert ratcheted["signal"].iloc[37:40].tolist() == [1, 0, 0]
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
