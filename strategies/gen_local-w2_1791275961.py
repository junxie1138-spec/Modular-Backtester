from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    percentile_window: int = 126
    entry_quantile: float = 0.85
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791275961"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(params.percentile_window + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.percentile_window < 3:
            raise ValueError("percentile_window must be at least 3")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["overnight"] = np.log(opening / closing.shift(1))
        out["intraday"] = np.log(closing / opening)
        g = out["overnight"]
        r = out["intraday"]
        # Second-order antisymmetric signature over three daily increments.
        out["area"] = 0.5 * (
            g.shift(2) * r.shift(1) - r.shift(2) * g.shift(1)
            + g.shift(2) * r - r.shift(2) * g
            + g.shift(1) * r - r.shift(1) * g
        )
        out["path_energy"] = (g.abs() + r.abs()).rolling(3, min_periods=3).sum()
        out["normalized_area"] = out["area"] / out["path_energy"].pow(2).replace(0.0, np.nan)
        out["area_threshold"] = out["normalized_area"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_quantile)
        out["monday"] = data.index.dayofweek == 0
        out["eligible"] = (
            out["monday"]
            & out["normalized_area"].notna()
            & out["area_threshold"].notna()
            & (out["normalized_area"] > out["area_threshold"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force a flat bar; entry conditions cannot postpone exit.
                    exit_bar = -1
            elif eligible[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    sample = pd.DataFrame(index=index)
    eligible = pd.DataFrame({"eligible": True}, index=index)
    for hold in (1, 2):
        result = GeneratedStrategy.generate_signals(
            sample, eligible, None, SignatureParams(hold_bars=hold)
        ).data
        expected = [0] + [int(i % (hold + 1) < hold) for i in range(6)]
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
