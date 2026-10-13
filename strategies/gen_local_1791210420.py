from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrengthParams:
    return_window: int = 20
    strength_threshold: float = 0.35
    atr_window: int = 14
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[StrengthParams]):
    strategy_id = "gen_local_1791210420"

    @classmethod
    def params_type(cls):
        return StrengthParams

    def warmup_bars(self, params: StrengthParams) -> int:
        return int(max(200, params.return_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrengthParams) -> pd.DataFrame:
        if params.return_window < 2 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; return_window must be at least two.")
        if not np.isfinite(params.strength_threshold) or params.strength_threshold <= 0:
            raise ValueError("strength_threshold must be finite and positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["negative_return_squared"] = out["return"].clip(upper=0).pow(2)
        out["mean_return"] = out["return"].rolling(params.return_window, min_periods=params.return_window).mean()
        out["downside_rms"] = np.sqrt(out["negative_return_squared"].rolling(params.return_window, min_periods=params.return_window).mean())
        out["strength"] = out["mean_return"] / out["downside_rms"].clip(lower=1e-8)
        out["previous_strength"] = out["strength"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = np.isfinite(out[["strength", "previous_strength", "ma200", "atr", "return"]]).all(axis=1)
        out["entry"] = (
            ready
            & (data["close"] > out["ma200"])
            & (out["return"] > 0)
            & (out["strength"] > params.strength_threshold)
            & (out["previous_strength"] <= params.strength_threshold)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrengthParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        held = 0
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # The preceding close's decision fills at this bar's open.
                # Freeze both the entry reference and the preceding bar's ATR.
                stop = opens[i] - params.stop_atr * atr[i - 1]
                active = bool(np.isfinite(stop) and np.isfinite(opens[i]) and opens[i] > 0)
                pending = False
                held = 0
                if not active:
                    continue
            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or held >= 2:
                    active = False
                    continue
                signals[i] = 1
            elif entries[i]:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"open": [100.0] * 4, "close": [100.0, 101.0, 102.0, 103.0]}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False], "atr": [1.0] * 4}, index=index)
    params = StrengthParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    data.loc[index[1], "close"] = 98.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    data.loc[index[1], ["open", "close"]] = [90.0, 89.0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
