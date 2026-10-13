from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    echo_strength: float = 0.15
    profit_target: float = 0.01


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1790993029"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return 98

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)

        # Fixed fractional order 0.6 with a causal 24-term filter.
        weights = np.empty(24, dtype=float)
        weights[0] = 1.0
        for k in range(1, 24):
            weights[k] = -weights[k - 1] * (0.6 - k + 1) / k
        fractional = close * weights[0]
        for k in range(1, 24):
            fractional = fractional + weights[k] * close.shift(k)

        baseline = fractional.rolling(32, min_periods=32).mean()
        pulse = fractional - baseline
        pulse_lag1 = pulse.shift(1)
        pulse_lag2 = pulse.shift(2)
        rho1 = pulse.rolling(40, min_periods=40).corr(pulse_lag1)
        rho2 = pulse.rolling(40, min_periods=40).corr(pulse_lag2)
        rho1 = rho1.replace([np.inf, -np.inf], np.nan)
        rho2 = rho2.replace([np.inf, -np.inf], np.nan)

        entry = (
            (rho1 < -params.echo_strength)
            & (rho2 > params.echo_strength)
            & (pulse < 0)
            & (pulse_lag1 > 0)
            & (pulse_lag2 < 0)
            & (pulse.abs() < pulse_lag2.abs())
        )
        return pd.DataFrame(
            {
                "fractional_close": fractional,
                "baseline": baseline,
                "pulse": pulse,
                "pulse_lag1": pulse_lag1,
                "pulse_lag2": pulse_lag2,
                "rho1": rho1,
                "rho2": rho2,
                "entry": entry.fillna(False),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        candidates = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                # The preceding close's entry decision fills at this open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0
                target = (
                    valid
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= 4
                if not valid or target or expired:
                    active = False
                else:
                    raw[i] = 1
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                raw[i] = 1

        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
