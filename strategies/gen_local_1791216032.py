from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 6
    area_threshold: float = 0.10
    energy_threshold: float = 0.35
    profit_target: float = 0.008
    max_hold_bars: int = 2

    def __post_init__(self):
        if self.lookback < 2 or self.max_hold_bars not in (1, 2):
            raise ValueError("lookback must be >= 2; max_hold_bars must be 1 or 2")
        if not (0 <= self.area_threshold < 1 and 0 <= self.energy_threshold < 1):
            raise ValueError("thresholds must be in [0, 1)")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791216032"

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff().replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        # Antisymmetric level-two signature of (unit time, log price).
        # Each return has weight 2*k-n-1, with k=1 for the oldest return.
        weighted = pd.Series(0.0, index=data.index)
        for lag in range(n):
            weighted = weighted + (n - 1 - 2 * lag) * r.shift(lag)
        out["signature_area"] = 0.5 * weighted
        out["absolute_path"] = r.abs().rolling(n, min_periods=n).sum()
        out["normalized_area"] = weighted / ((n - 1) * out["absolute_path"].replace(0, np.nan))
        out["up_energy"] = r.clip(lower=0).pow(2).rolling(n, min_periods=n).sum()
        out["down_energy"] = r.clip(upper=0).pow(2).rolling(n, min_periods=n).sum()
        total = out["up_energy"] + out["down_energy"]
        out["energy_balance"] = (out["up_energy"] - out["down_energy"]) / total.replace(0, np.nan)
        out["entry"] = (
            out["normalized_area"].gt(params.area_threshold)
            & out["energy_balance"].gt(params.energy_threshold)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                # Next-session open is the fill-price proxy; exits are close decisions.
                entry_price = opens[i]
                entry_bar = i
            if entry_bar >= 0:
                age = i - entry_bar + 1
                hit_target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                if hit_target or age >= params.max_hold_bars:
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SignatureParams(profit_target=0.005)
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 89.0,
        "close": [90.0, 101.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    forced = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
