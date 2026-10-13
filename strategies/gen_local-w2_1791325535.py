from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalShockParams:
    pattern_window: int = 12
    complexity_jump: float = 0.10
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[OrdinalShockParams]):
    strategy_id = "gen_local-w2_1791325535"

    @classmethod
    def params_type(cls) -> type[OrdinalShockParams]:
        return OrdinalShockParams

    @staticmethod
    def warmup_bars(params: OrdinalShockParams) -> int:
        return 2 * max(3, int(params.pattern_window)) + 4

    def indicators(self, data: pd.DataFrame, params: OrdinalShockParams) -> pd.DataFrame:
        window = max(3, int(params.pattern_window))
        result = pd.DataFrame(index=data.index)
        result["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        result["return_lag1"] = result["return"].shift(1)
        result["return_lag2"] = result["return"].shift(2)
        triples = result[["return_lag2", "return_lag1", "return"]].to_numpy()
        order = np.argsort(triples, axis=1, kind="stable")
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        valid = np.isfinite(triples).all(axis=1)
        result["ordinal_pattern"] = pd.Series(codes, index=data.index, dtype=float).where(valid)
        result["previous_pattern"] = result["ordinal_pattern"].shift(1)
        result["ordinal_transition"] = result["previous_pattern"] * 27 + result["ordinal_pattern"]
        collision = pd.Series(0.0, index=data.index)
        for previous in (5, 7, 11, 15, 19, 21):
            for current in (5, 7, 11, 15, 19, 21):
                name = f"transition_probability_{previous}_{current}"
                occurrence = result["ordinal_transition"].eq(previous * 27 + current).astype(float)
                occurrence = occurrence.where(result["ordinal_transition"].notna())
                result[name] = occurrence.rolling(window, min_periods=window).mean()
                collision = collision + result[name].pow(2)
        result["complexity"] = 1.0 - collision
        result["previous_complexity"] = result["complexity"].shift(window)
        result["complexity_change"] = result["complexity"] - result["previous_complexity"]
        result["drift"] = result["return"].rolling(window, min_periods=window).mean()
        result["previous_drift"] = result["drift"].shift(window)
        required = ["complexity_change", "drift", "previous_drift", "return"]
        ready = result[required].notna().all(axis=1)
        result["candidate"] = (
            ready
            & result["complexity_change"].ge(params.complexity_jump)
            & result["previous_drift"].lt(0.0)
            & result["drift"].gt(0.0)
            & result["return"].gt(0.0)
        )
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalShockParams,
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(False).astype(bool)
        confirmed = candidate & candidate.shift(1, fill_value=False)
        holding = max(1, int(params.holding_bars))
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for bar, enter in enumerate(confirmed.to_numpy(dtype=bool)):
            if entry_bar >= 0:
                if bar - entry_bar >= holding:
                    entry_bar = -1
                else:
                    signal[bar] = 1
            elif enter:
                entry_bar = bar
                signal[bar] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=16, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"candidate": [False, True, False, True, True] + [False] * 11}, index=index)
    params = OrdinalShockParams(holding_bars=8)
    output = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    expected = [0] * 5 + [1] * 8 + [0] * 3
    assert output["signal"].tolist() == expected
    assert output["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
