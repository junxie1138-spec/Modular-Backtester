from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    return_window: int = 20
    range_window: int = 5
    disagreement_threshold: float = 1.5
    target_vol: float = 0.10
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791332958"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.return_window + 1, params.range_window))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.return_window < 2 or params.range_window < 2:
            raise ValueError("Volatility windows must be at least two bars.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        if not np.isfinite(params.disagreement_threshold) or params.disagreement_threshold <= 1:
            raise ValueError("disagreement_threshold must be finite and greater than one.")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        high = data["high"]
        low = data["low"]
        valid_range = (high >= low) & (low > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["log_range"] = np.log(high.where(valid_range) / low.where(valid_range)).replace([np.inf, -np.inf], np.nan)
        out["range_variance"] = out["log_range"].pow(2) / (4.0 * np.log(2.0))
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["close_vol"] = out["log_return"].rolling(params.return_window, min_periods=params.return_window).std(ddof=1)
        out["range_vol"] = np.sqrt(out["range_variance"].rolling(params.range_window, min_periods=params.range_window).mean())
        out["disagreement"] = out["range_vol"] / out["close_vol"].where(out["close_vol"] > 0)
        out["risk_vol"] = out[["close_vol", "range_vol"]].max(axis=1, skipna=False) * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["risk_vol"].where(out["risk_vol"] > 0)).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95)
        out["previous_disagreement"] = out["disagreement"].shift(1)
        ready = np.isfinite(out[["ma200", "disagreement", "previous_disagreement", "target_size"]]).all(axis=1)
        out["entry"] = (
            ready
            & (close > out["ma200"])
            & (out["disagreement"] > params.disagreement_threshold)
            & (out["previous_disagreement"] <= params.disagreement_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        targets = indicators["target_size"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        exit_at = -1
        held_size = 0.95

        for i in range(len(data)):
            if exit_at >= 0:
                if i == exit_at:
                    # Force a flat bar; do not replace the timed exit with a new entry.
                    exit_at = -1
                else:
                    signal[i] = 1
                    sizes[i] = held_size
            elif entries[i] and np.isfinite(targets[i]) and targets[i] > 0:
                exit_at = i + params.hold_bars
                held_size = float(np.clip(targets[i], 0.05, 0.95))
                signal[i] = 1
                sizes[i] = held_size

        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": True, "target_size": np.arange(8) * 0.01 + 0.3}, index=index)
    result = strategy.generate_signals(data, indicators, None, GeneratedParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert result["size"].iloc[1] == result["size"].iloc[2] == 0.3
    one_bar = strategy.generate_signals(data, indicators, None, GeneratedParams(hold_bars=1)).data
    assert one_bar["signal"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    warmup = strategy.indicators(data, GeneratedParams())
    assert not warmup["entry"].any()
    assert strategy.generate_signals(data, warmup, None, GeneratedParams()).data["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
