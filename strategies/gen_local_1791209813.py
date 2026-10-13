from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalTransitionParams:
    complexity_window: int = 42
    percentile_window: int = 252
    momentum_bars: int = 15
    regime_tail: float = 0.25
    momentum_percentile: float = 0.65
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[OrdinalTransitionParams]):
    strategy_id = "gen_local_1791209813"

    @classmethod
    def params_type(cls) -> type[OrdinalTransitionParams]:
        return OrdinalTransitionParams

    def warmup_bars(self, params: OrdinalTransitionParams) -> int:
        return int(max(
            params.complexity_window + params.percentile_window + 4,
            params.momentum_bars + params.percentile_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: OrdinalTransitionParams) -> pd.DataFrame:
        if min(params.complexity_window, params.percentile_window, params.momentum_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.regime_tail < 0.5:
            raise ValueError("regime_tail must lie between zero and one half")
        if not 0.0 < params.momentum_percentile < 1.0:
            raise ValueError("momentum_percentile must lie between zero and one")
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError("Exit parameters must be positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = returns
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns retain chronological order, giving deterministic patterns.
        conditions = [
            (a <= b) & (b <= c),
            (a <= c) & (c < b),
            (b < a) & (a <= c),
            (b <= c) & (c < a),
            (c < a) & (a <= b),
            (c < b) & (b < a),
        ]
        pattern = pd.Series(np.select(conditions, range(6), default=np.nan), index=data.index).where(valid)
        out["pattern"] = pattern
        transition = pattern.shift(1) * 6.0 + pattern
        out["transition"] = transition
        transition_valid = transition.notna()
        joint_square = pd.Series(0.0, index=data.index)
        source_square = pd.Series(0.0, index=data.index)
        for source in range(6):
            source_count = pd.Series(0.0, index=data.index)
            for destination in range(6):
                code = source * 6 + destination
                count = transition.eq(code).astype(float).where(transition_valid).rolling(
                    params.complexity_window, min_periods=params.complexity_window
                ).sum()
                out[f"transition_count_{code}"] = count
                joint_square = joint_square + count.pow(2)
                source_count = source_count + count
            out[f"source_count_{source}"] = source_count
            source_square = source_square + source_count.pow(2)
        out["joint_collision_count"] = joint_square
        out["source_collision_count"] = source_square
        complexity = (1.0 - joint_square / source_square.where(source_square > 0.0)).clip(0.0, 1.0)
        out["complexity"] = complexity
        history = complexity.shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out["complexity_low"] = history.quantile(params.regime_tail)
        out["complexity_high"] = history.quantile(1.0 - params.regime_tail)
        regime = pd.Series(np.nan, index=data.index)
        separated = out["complexity_high"] > out["complexity_low"]
        regime.loc[separated & complexity.le(out["complexity_low"])] = 1.0
        regime.loc[separated & complexity.ge(out["complexity_high"])] = -1.0
        out["regime"] = regime.ffill().fillna(0.0)
        momentum = close.pct_change(params.momentum_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["momentum"] = momentum
        out["momentum_magnitude"] = momentum.abs()
        out["momentum_threshold"] = momentum.abs().shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.momentum_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalTransitionParams,
    ) -> SignalFrame:
        eligible = (
            indicators["complexity"].notna()
            & indicators["complexity_low"].notna()
            & indicators["complexity_high"].notna()
            & indicators["momentum_threshold"].notna()
            & indicators["momentum_magnitude"].gt(indicators["momentum_threshold"])
        )
        candidate = (np.sign(indicators["momentum"]) * indicators["regime"]).where(
            eligible, 0.0
        ).fillna(0.0).astype(int).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                # Yesterday's decision fills at today's open.
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        decisions[i] = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_close else -np.inf
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if gain >= params.profit_target or expired:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(candidate[i])
            decisions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
