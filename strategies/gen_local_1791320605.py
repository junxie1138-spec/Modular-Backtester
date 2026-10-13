from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 20
    pressure_window: int = 18
    entropy_cutoff: float = 0.90
    ordinal_edge: float = 0.10
    pressure_edge: float = 0.05


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791320605"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.pattern_window + 2, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.pattern_window < 3 or params.pressure_window < 1:
            raise ValueError("Windows must be positive; pattern_window must be at least 3.")
        if not 0.0 < params.entropy_cutoff < 1.0:
            raise ValueError("entropy_cutoff must lie between zero and one.")
        if not 0.0 < params.ordinal_edge < 1.0 or not 0.0 < params.pressure_edge < 1.0:
            raise ValueError("Direction thresholds must lie between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["old_close"] = data["close"].shift(2)
        out["middle_close"] = data["close"].shift(1)
        a, b, c = out["old_close"], out["middle_close"], data["close"]
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        # Equal closes are ranked by time: the later observation ranks higher.
        out["old_rank"] = ((a > b).astype(int) + (a > c).astype(int)).where(valid)
        out["middle_rank"] = ((b >= a).astype(int) + (b > c).astype(int)).where(valid)
        out["ordinal_code"] = 3 * out["old_rank"] + out["middle_rank"]
        out["entropy"] = 0.0
        for code in (1, 2, 3, 5, 6, 7):
            name = "prob_" + str(code)
            out[name] = (
                out["ordinal_code"].eq(code).astype(float).where(valid)
                .rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            )
            out["entropy"] -= out[name] * np.log(out[name].clip(lower=1e-12)) / np.log(6.0)
        out["ordinal_bias"] = out["prob_1"] - out["prob_7"]
        out["ordinal_direction"] = out["ordinal_bias"].where(
            out["entropy"] <= params.entropy_cutoff, -out["ordinal_bias"]
        )
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] >= data["low"]
        )
        out["close_location"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / out["range"].replace(0.0, np.nan)
        ).clip(-1.0, 1.0)
        out.loc[out["range"].eq(0.0), "close_location"] = 0.0
        out["valid_volume"] = data["volume"].where(
            np.isfinite(data["volume"]) & data["volume"].ge(0.0)
        )
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["pressure_sum"] = out["signed_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["pressure"] = out["pressure_sum"] / out["volume_sum"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        ready = np.isfinite(indicators[["entropy", "ordinal_direction", "pressure"]]).all(axis=1)
        long_entry = ready & indicators["ordinal_direction"].gt(params.ordinal_edge) & indicators["pressure"].gt(params.pressure_edge)
        short_entry = ready & indicators["ordinal_direction"].lt(-params.ordinal_edge) & indicators["pressure"].lt(-params.pressure_edge)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Exit to flat on opposite AND agreement; neutral readings hold.
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    direction = [0.3, 0.0, -0.3, -0.3, np.nan, 0.3, 0.3]
    ind = pd.DataFrame(
        {"entropy": 0.8, "ordinal_direction": direction, "pressure": direction},
        index=index,
    )
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= params.pattern_window + 2
    prices = np.arange(1.0, 61.0)
    bars = pd.DataFrame(
        {"open": prices, "high": prices + 1.0, "low": prices - 1.0,
         "close": prices, "volume": 100.0},
        index=pd.date_range("2020-01-01", periods=60),
    )
    derived = strategy.indicators(bars, params)
    assert derived["entropy"].iloc[:params.pattern_window + 1].isna().all()
    assert np.isclose(derived["prob_1"].iloc[-1], 1.0)
    assert np.isclose(derived["entropy"].iloc[-1], 0.0)
    assert derived.index.equals(bars.index)


if __name__ == "__main__":
    _self_check()
