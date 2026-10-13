from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalFlowParams:
    pattern_window: int = 48
    complexity_min: float = 0.20
    diversity_min: float = 0.65
    regime_window: int = 200
    volatility_window: int = 20
    target_volatility: float = 0.12
    atr_window: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalFlowParams]):
    strategy_id = "gen_local_1790988129"

    @classmethod
    def params_type(cls) -> type[OrdinalFlowParams]:
        return OrdinalFlowParams

    def warmup_bars(self, params: OrdinalFlowParams) -> int:
        return int(max(params.regime_window, params.pattern_window + 3,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalFlowParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        body = close.div(data["open"].replace(0.0, np.nan)).sub(1.0)
        a, b, c = body.shift(2), body.shift(1), body
        valid = pd.concat([a, b, c], axis=1).notna().all(axis=1)
        pattern = pd.Series(np.select(
            [(a <= b) & (b <= c), (a <= c) & (c < b),
             (b < a) & (a <= c), (b <= c) & (c < a),
             (c < a) & (a <= b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=0),
            index=data.index, dtype=float).where(valid)
        previous = pattern.shift(1)
        pair = previous.mul(6).add(pattern)
        window = params.pattern_window
        pair_valid = pair.notna()
        prior_entropy = pd.Series(0.0, index=data.index)
        joint_entropy = pd.Series(0.0, index=data.index)
        for state in range(6):
            probability = previous.eq(state).astype(float).where(pair_valid).rolling(window).mean()
            prior_entropy -= probability * np.log(probability.clip(lower=1e-12))
        for transition in range(36):
            probability = pair.eq(transition).astype(float).where(pair_valid).rolling(window).mean()
            joint_entropy -= probability * np.log(probability.clip(lower=1e-12))
        diversity = prior_entropy.div(np.log(6.0)).clip(0.0, 1.0)
        conditional = (joint_entropy - prior_entropy).div(np.log(6.0)).clip(0.0, 1.0)
        returns = close.pct_change(fill_method=None)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["body"] = body
        out["previous_body"] = body.shift(1)
        out["return"] = returns
        out["ordinal_pattern"] = pattern
        out["previous_pattern"] = previous
        out["ordinal_transition"] = pair
        out["prior_entropy"] = prior_entropy
        out["joint_entropy"] = joint_entropy
        out["diversity"] = diversity
        out["conditional_entropy"] = conditional
        out["complexity"] = diversity * (1.0 - conditional)
        out["regime_ma"] = close.rolling(params.regime_window).mean()
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_window).mean()
        out["annual_volatility"] = returns.rolling(params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalFlowParams) -> SignalFrame:
        ind = indicators
        ready = ind[["complexity", "diversity", "regime_ma", "atr",
                     "annual_volatility", "body", "previous_body", "return"]].notna().all(axis=1)
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        candidate = (ready & (data["close"] > ind["regime_ma"])
                     & (ind["diversity"] >= params.diversity_min)
                     & (ind["complexity"] >= params.complexity_min)
                     & (ind["previous_body"] < 0.0)
                     & (ind["body"] > 0.0) & (ind["return"] > 0.0))
        size = (params.target_volatility / ind["annual_volatility"].replace(0.0, np.nan))
        size = size.replace([np.inf, -np.inf], np.nan).clip(0.10, 0.95).fillna(0.10)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = ind["atr"].to_numpy(dtype=float)
        ma = ind["regime_ma"].to_numpy(dtype=float)
        entries = candidate.fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = stop = 0.0
        age = 0
        max_age = min(5, max(3, int(params.holding_bars)))
        for i in range(len(data)):
            if pending:
                pending = False
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0.0 or not np.isfinite(atr[i - 1]):
                    continue
                entry = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                if (not np.isfinite(closes[i]) or not np.isfinite(lows[i])
                        or not np.isfinite(atr[i]) or not np.isfinite(ma[i])
                        or lows[i] <= stop or closes[i] <= ma[i] or age >= max_age):
                    active = False
                    continue
                # Test the previously committed stop before raising it at this close.
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, closes[i] - params.trail_atr * atr[i])
                raw[i] = 1
            elif entries[i]:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": size.shift(1).fillna(0.10)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
