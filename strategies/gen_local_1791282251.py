from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    baseline_bars: int = 40
    recent_bars: int = 7
    change_threshold: float = 1.8
    reset_threshold: float = 0.5
    profit_target: float = 0.025
    max_hold_bars: int = 8

    def __post_init__(self):
        if self.baseline_bars < 2 or self.recent_bars < 2:
            raise ValueError("Return windows must contain at least two bars.")
        if not 0 <= self.reset_threshold < self.change_threshold:
            raise ValueError("Require 0 <= reset_threshold < change_threshold.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")
        if self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791282251"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.baseline_bars + params.recent_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        returns = data["close"].pct_change(fill_method=None)
        up = returns.gt(0).astype(float).where(returns.notna())
        recent = up.rolling(params.recent_bars).mean()
        baseline = up.shift(params.recent_bars).rolling(params.baseline_bars).mean()
        pooled = (
            recent * params.recent_bars + baseline * params.baseline_bars
        ) / (params.recent_bars + params.baseline_bars)
        variance = pooled * (1 - pooled) * (
            1 / params.recent_bars + 1 / params.baseline_bars
        )
        change = (recent - baseline) / np.sqrt(variance.clip(lower=1e-12))
        return pd.DataFrame(
            {
                "return": returns,
                "up": up,
                "recent_up": recent,
                "baseline_up": baseline,
                "pooled_up": pooled,
                "sign_variance": variance,
                "change": change,
                "ma200": data["close"].rolling(200).mean(),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        entry = (
            data["close"].gt(indicators["ma200"])
            & indicators["baseline_up"].le(0.5)
            & indicators["recent_up"].ge(0.7)
            & indicators["change"].ge(params.change_threshold)
        ).fillna(False).to_numpy()
        reset = indicators["change"].le(params.reset_threshold).fillna(False).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        if not np.isfinite(opens).all() or not np.isfinite(closes).all():
            raise ValueError("Open and close prices must be finite.")
        if (opens <= 0).any() or (closes <= 0).any():
            raise ValueError("Open and close prices must be positive.")
        decisions = np.zeros(len(data), dtype=int)
        active = False
        armed = True
        entry_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            if reset[i]:
                armed = True
            if active:
                # An entry decided yesterday fills at today's open.
                if entry_price == 0.0:
                    entry_price = opens[i]
                    entry_bar = i
                target_hit = closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    active = False
                    entry_price = 0.0
                    # Keep this close's exit decision flat; no same-bar reentry.
                    continue
                decisions[i] = 1
            elif armed and entry[i]:
                active = True
                armed = False
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=210)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    strategy = GeneratedStrategy()
    params = Params(max_hold_bars=2)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.warmup_bars(params) >= 200
    assert derived["baseline_up"].iloc[:47].isna().all()
    evidence = pd.DataFrame(
        {"ma200": 99.0, "baseline_up": 0.4, "recent_up": 0.8, "change": 0.0},
        index=index,
    )
    evidence.loc[index[200], "change"] = 3.0
    timed = strategy.generate_signals(data, evidence, None, params).data
    assert timed["signal"].iloc[200:204].tolist() == [0, 1, 1, 0]
    data.loc[index[201], "close"] = 103.0
    targeted = strategy.generate_signals(data, evidence, None, params).data
    assert targeted["signal"].iloc[200:204].tolist() == [0, 1, 0, 0]
    assert targeted["signal"].isin([0, 1]).all()
    assert targeted["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
