from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapResponseParams:
    gap_window: int = 126
    response_window: int = 40
    regime_window: int = 126
    gap_percentile: float = 0.85
    regime_percentile: float = 0.65
    profit_target: float = 0.06
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapResponseParams]):
    strategy_id = "gen_local-w2_1791318411"

    @classmethod
    def params_type(cls) -> type[GapResponseParams]:
        return GapResponseParams

    @staticmethod
    def warmup_bars(params: GapResponseParams) -> int:
        return int(max(params.gap_window + 2,
                       params.response_window + params.regime_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapResponseParams) -> pd.DataFrame:
        if min(params.gap_window, params.response_window, params.regime_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.gap_percentile < 1.0:
            raise ValueError("gap_percentile must be between zero and one")
        if not 0.5 < params.regime_percentile < 1.0:
            raise ValueError("regime_percentile must be between 0.5 and one")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        out["gap"] = (opening / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = (data["close"].where(lambda x: x > 0) / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_threshold"] = out["gap"].abs().shift(1).rolling(params.gap_window, min_periods=params.gap_window).quantile(params.gap_percentile)
        out["response"] = out["gap"].rolling(params.response_window, min_periods=params.response_window).corr(out["intraday"]).replace([np.inf, -np.inf], np.nan)
        history = out["response"].shift(1).rolling(params.regime_window, min_periods=params.regime_window)
        out["response_upper"] = history.quantile(params.regime_percentile)
        out["response_lower"] = history.quantile(1.0 - params.regime_percentile)
        continuation = (out["response"] > out["response_upper"]) & (out["response"] > 0)
        reversal = (out["response"] < out["response_lower"]) & (out["response"] < 0)
        out["regime"] = np.select([continuation, reversal], [1, -1], default=0).astype(int)
        out["direction"] = np.sign(out["gap"].fillna(0)).astype(int) * out["regime"]
        ready = out[["gap", "intraday", "gap_threshold", "response", "response_upper", "response_lower"]].notna().all(axis=1)
        extreme = out["gap"].abs() > out["gap_threshold"]
        confirmation = out["intraday"] * out["direction"] > 0
        out["entry"] = out["direction"].where(ready & extreme & confirmation, 0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapResponseParams) -> SignalFrame:
        if params.holding_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Holding period and profit target must be positive")
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction = 0
        pending = 0
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # A previous close's decision fills at this bar's open.
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    direction = pending
                    entry_price = opens[i]
                    entry_bar = i
                pending = 0
            if direction:
                gain = direction * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.holding_bars
                if target_hit or time_hit:
                    direction = 0
                    continue
                signals[i] = direction
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                pending = int(entries[i])
                signals[i] = pending
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    for side, closing, limit in [(1, 107.0, 18), (-1, 93.0, 18), (1, 100.0, 2)]:
        data = pd.DataFrame({"open": 100.0, "high": 108.0, "low": 92.0,
                             "close": [100.0, 100.0, closing, 100.0, 100.0, 100.0],
                             "volume": 1000000.0}, index=index)
        indicators = pd.DataFrame({"entry": [side, 0, 0, 0, 0, 0]}, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None,
                    GapResponseParams(holding_bars=limit)).data
        assert result["signal"].tolist() == [0, side, side, 0, 0, 0]
        assert (result["size"] > 0).all()
