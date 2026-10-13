from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    z_threshold: float = 1.5
    impact_threshold: float = 0.75
    trail_atr: float = 2.5
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local-w2_1791300343"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return 90

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        ret = data["close"].pct_change()
        volatility = ret.rolling(63).std() * np.sqrt(15)
        z = ret.rolling(15).sum() / volatility.replace(0, np.nan)

        relative_volume = data["volume"] / data["volume"].rolling(20).median().replace(0, np.nan)
        impact = ret.abs().rolling(5).sum() / relative_volume.rolling(5).sum().replace(0, np.nan)
        impact_ratio = impact / impact.rolling(63).median().replace(0, np.nan)

        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()

        return pd.DataFrame({
            "z": z.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "impact_ratio": impact_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0),
            "atr": atr.fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AbsorptionParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        impact_ratio = indicators["impact_ratio"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        extreme = 0.0
        stop = 0.0
        held = 0

        for i in range(self.warmup_bars(params), len(data)):
            if position:
                held += 1
                if position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                    if close[i] <= stop or held >= params.max_hold:
                        position = 0
                else:
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                    if close[i] >= stop or held >= params.max_hold:
                        position = 0
            elif atr[i] > 0 and impact_ratio[i] < params.impact_threshold:
                if z[i] <= -params.z_threshold:
                    position = 1
                elif z[i] >= params.z_threshold:
                    position = -1
                if position:
                    extreme = close[i]
                    stop = close[i] - position * params.trail_atr * atr[i]
                    held = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
