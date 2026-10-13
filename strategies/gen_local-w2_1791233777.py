from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_window: int = 20
    innovation_window: int = 20
    evidence_window: int = 5
    evidence_threshold: float = 0.6
    regime_buffer: float = 0.005
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791233777"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.atr_window + 1,
                       params.slope_window + params.innovation_window
                       + params.evidence_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.slope_window < 2 or params.innovation_window < 2
                or params.evidence_window < 1 or params.atr_window < 1
                or params.hold_bars < 1 or params.stop_atr <= 0
                or not 0 <= params.regime_buffer < 1
                or not 0 < params.evidence_threshold < 2):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        n = params.slope_window
        position = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["log_sum"] = out["log_close"].rolling(n, min_periods=n).sum()
        out["weighted_log_sum"] = (position * out["log_close"]).rolling(n, min_periods=n).sum()
        denominator = n * (n * n - 1) / 12.0
        out["latent_slope"] = (
            out["weighted_log_sum"]
            - (position - (n - 1) / 2.0) * out["log_sum"]
        ) / denominator
        out["innovation"] = out["latent_slope"].diff()
        out["innovation_scale"] = out["innovation"].rolling(
            params.innovation_window, min_periods=params.innovation_window
        ).std(ddof=0).shift(1)
        out["standardized_innovation"] = (
            out["innovation"] / out["innovation_scale"].where(out["innovation_scale"] > 1e-12)
        ).replace([np.inf, -np.inf], np.nan)
        out["bounded_innovation"] = out["standardized_innovation"].clip(-2.0, 2.0)
        out["evidence"] = out["bounded_innovation"].rolling(
            params.evidence_window, min_periods=params.evidence_window
        ).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["regime_on"] = close > out["ma200"] * (1.0 + params.regime_buffer)
        out["regime_off"] = close < out["ma200"] * (1.0 - params.regime_buffer)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["entry"] = (
            out["ma200"].notna()
            & (out["latent_slope"] > 0)
            & (out["evidence"] > params.evidence_threshold)
            & (out["evidence"].shift(1) <= params.evidence_threshold)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        regime_on = indicators["regime_on"].fillna(False).to_numpy(dtype=bool)
        regime_off = indicators["regime_off"].fillna(False).to_numpy(dtype=bool)
        bullish = False
        active = False
        pending_entry = False
        entry_atr = 0.0
        fixed_stop = 0.0
        age = 0
        for i in range(len(data)):
            if regime_off[i]:
                bullish = False
            elif regime_on[i]:
                bullish = True
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                    age = 0
                else:
                    active = False
            if active:
                age += 1
                if (not np.isfinite(closes[i]) or closes[i] < fixed_stop
                        or age >= params.hold_bars or not bullish):
                    active = False
                else:
                    decisions[i] = 1
            elif (bullish and entries[i] and np.isfinite(closes[i])
                  and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                entry_atr = atr[i]
                active = True
                pending_entry = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    sample = pd.DataFrame({
        "open": [100.0] * 7,
        "close": [100.0, 101.0, 99.0, 97.0, 100.0, 100.0, 100.0],
    }, index=index)
    derived = pd.DataFrame({
        "atr": 1.0,
        "entry": [True, False, False, False, False, False, False],
        "regime_on": True,
        "regime_off": False,
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, sample, derived, None, StrategyParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert result.index.equals(index) and (result["size"] > 0).all()
