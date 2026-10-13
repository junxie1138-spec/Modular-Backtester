from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalTransitionParams:
    complexity_window: int = 36
    entropy_cutoff: float = 0.55
    center_window: int = 12
    atr_window: int = 20
    tension_threshold: float = 1.25
    holding_bars: int = 7


class GeneratedStrategy(BaseStrategy[OrdinalTransitionParams]):
    strategy_id = "gen_local_1791331931"

    @classmethod
    def params_type(cls):
        return OrdinalTransitionParams

    @staticmethod
    def warmup_bars(params: OrdinalTransitionParams) -> int:
        return int(max(params.complexity_window + 4, params.center_window, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalTransitionParams) -> pd.DataFrame:
        if min(params.complexity_window, params.center_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError("Window lengths and holding_bars must be positive")
        if not 0.0 < params.entropy_cutoff < 1.0 or params.tension_threshold <= 0.0:
            raise ValueError("Invalid entropy cutoff or tension threshold")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        a, b, c = out["return_lag2"], out["return_lag1"], out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns retain chronological order, giving deterministic ties.
        out["pattern"] = ((a > b).astype(int) + 2 * (a > c).astype(int) + 4 * (b > c).astype(int)).where(valid)
        out["previous_pattern"] = out["pattern"].shift(1)
        pair_valid = out["pattern"].notna() & out["previous_pattern"].notna()
        window = params.complexity_window
        codes = (0, 1, 3, 4, 6, 7)
        entropy = pd.Series(0.0, index=data.index)
        for previous in codes:
            prior_name = f"prior_{previous}"
            out[prior_name] = out["previous_pattern"].eq(previous).astype(float).where(pair_valid).rolling(window, min_periods=window).mean()
            for current in codes:
                name = f"transition_{previous}_{current}"
                out[name] = (out["previous_pattern"].eq(previous) & out["pattern"].eq(current)).astype(float).where(pair_valid).rolling(window, min_periods=window).mean()
                probability = out[name]
                conditional = probability / out[prior_name].clip(lower=1e-12)
                entropy = entropy - probability * np.log(conditional.clip(lower=1e-12))
        out["complexity"] = (entropy / np.log(6.0)).clip(lower=0.0, upper=1.0)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["center"] = data["close"].rolling(params.center_window, min_periods=params.center_window).mean()
        out["tension"] = ((data["close"] - out["center"]) / out["atr"].where(out["atr"] > 0.0)).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalTransitionParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        complexity = indicators["complexity"]
        tension = indicators["tension"]
        ready = complexity.notna() & tension.notna()
        trend_regime = complexity.lt(params.entropy_cutoff)
        stretched = tension.abs().ge(params.tension_threshold)
        direction = np.sign(tension.fillna(0.0)).astype(int)
        candidate = direction.where(trend_regime, -direction).where(ready & stretched, 0).astype(int)
        confirmed = candidate.ne(0) & candidate.eq(candidate.shift(1)) & trend_regime.eq(trend_regime.shift(1))
        entries = candidate.where(confirmed, 0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0:
                if bar - entry_bar >= params.holding_bars:
                    position = 0
            elif entry != 0:
                position = int(entry)
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = OrdinalTransitionParams()
    index = pd.date_range("2020-01-01", periods=20)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"complexity": 0.1, "tension": 2.0}, index=index)
    features.loc[index[3]:, "tension"] = -2.0
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:9].eq(1).all()
    assert result["signal"].iloc[9] == 0
    assert result["signal"].iloc[10:17].eq(-1).all()
    assert result["signal"].iloc[17] == 0
    assert result["size"].gt(0.0).all()
    calculated = GeneratedStrategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert calculated["tension"].isna().all()
    assert GeneratedStrategy.warmup_bars(params) >= params.complexity_window + 4


if __name__ == "__main__":
    _self_check()
