from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockQuarantineParams:
    shock_multiple: float = 2.2
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[ShockQuarantineParams]):
    strategy_id = "gen_local_1790996153"

    @classmethod
    def params_type(cls) -> type[ShockQuarantineParams]:
        return ShockQuarantineParams

    def warmup_bars(self, params: ShockQuarantineParams) -> int:
        return 25

    def indicators(self, data: pd.DataFrame, params: ShockQuarantineParams) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(20, min_periods=20).mean()
        baseline = atr.shift(1).replace(0.0, np.nan)
        shock = tr.gt(params.shock_multiple * baseline) & baseline.notna()
        result = pd.DataFrame(index=data.index)
        result["atr"] = atr
        result["shock_range"] = tr.shift(3)
        result["shock_baseline"] = baseline.shift(3)
        result["anchor"] = close.shift(4)
        result["quiet_range"] = tr.rolling(3, min_periods=3).mean()
        result["quarantined_shock"] = shock.shift(3, fill_value=False)
        result["fresh_shocks"] = shock.astype(int).rolling(3, min_periods=3).sum()
        result["displacement"] = (close - result["anchor"]) / result["shock_baseline"].replace(0.0, np.nan)
        ready = result[["atr", "shock_range", "shock_baseline", "anchor", "quiet_range", "fresh_shocks", "displacement"]].notna().all(axis=1)
        ready &= result["atr"].gt(0.0) & close.gt(0.0)
        eligible = ready & result["quarantined_shock"]
        eligible &= result["fresh_shocks"].eq(0)
        eligible &= result["quiet_range"].lt(0.65 * result["shock_range"])
        result["entry"] = np.select([
            eligible & result["displacement"].le(-0.8),
            eligible & result["displacement"].ge(0.8),
        ], [1, -1], default=0).astype(int)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockQuarantineParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["entry"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        entry_bar = -1
        stop = -np.inf
        armed = False
        cooldown_until = -1
        for i in range(len(data)):
            if side != 0 and np.isnan(entry):
                # Yesterday's decision fills at today's open.
                entry = opens[i]
                entry_bar = i
                if not np.isfinite(entry) or entry <= 0.0 or i == 0 or not np.isfinite(atrs[i - 1]) or atrs[i - 1] <= 0.0:
                    side = 0
                    entry = np.nan
                    cooldown_until = i + 5
                else:
                    # Signed price makes both long and short stops ratchet upward.
                    stop = side * entry - 3.0 * atrs[i - 1]
                    armed = False
            if side != 0:
                close = closes[i]
                if not np.isfinite(close) or close <= 0.0:
                    signals[i] = side
                    continue
                signed_close = side * close
                expired = i - entry_bar + 1 >= 20
                # Close-based exits use the stop established on earlier closes.
                if signed_close <= stop or expired:
                    side = 0
                    entry = np.nan
                    armed = False
                    stop = -np.inf
                    cooldown_until = i + 5
                else:
                    if side * (close / entry - 1.0) >= 0.01:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                            stop = max(stop, signed_close - params.trail_atr * atrs[i])
            elif i > cooldown_until and candidates[i] != 0 and i + 1 < len(data):
                side = int(candidates[i])
                entry = np.nan
            signals[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        assert df["signal"].isin([-1, 0, 1]).all()
        assert df["size"].gt(0.0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
