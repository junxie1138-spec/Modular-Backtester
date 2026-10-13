from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReversalParams:
    breakout_lookback: int = 15
    gap_lookback: int = 20
    gap_multiple: float = 0.75
    close_location_min: float = 0.75
    profit_target: float = 0.06
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapReversalParams]):
    strategy_id = "gen_local-w2_1791336434"

    @classmethod
    def params_type(cls) -> type[GapReversalParams]:
        return GapReversalParams

    def warmup_bars(self, params: GapReversalParams) -> int:
        if params.breakout_lookback < 2 or params.gap_lookback < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        if not np.isfinite(params.gap_multiple) or params.gap_multiple <= 0:
            raise ValueError("gap_multiple must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not 0 <= params.close_location_min <= 1:
            raise ValueError("close_location_min must be between zero and one")
        return int(max(params.breakout_lookback + 1, params.gap_lookback + 2))

    def indicators(self, data: pd.DataFrame, params: GapReversalParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = data["open"] / out["prior_close"] - 1.0
        out["gap_scale"] = out["gap"].abs().rolling(
            params.gap_lookback, min_periods=params.gap_lookback
        ).mean().shift(1)
        out["prior_high"] = data["high"].rolling(
            params.breakout_lookback, min_periods=params.breakout_lookback
        ).max().shift(1)
        out["day_range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["close_location"] = (data["close"] - data["low"]) / out["day_range"]
        out["gap_intensity"] = -out["gap"] / out["gap_scale"].where(lambda s: s > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["strength"] = (out["gap_intensity"] / 3.0).clip(0.25, 1.0).fillna(0.25)
        out["entry"] = (
            (out["gap"] < 0)
            & (out["gap_intensity"] >= params.gap_multiple)
            & (data["close"] > out["prior_high"])
            & (out["close_location"] >= params.close_location_min)
            & (data["open"] > 0)
            & (data["close"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReversalParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].to_numpy(dtype=float)
        pending_entry = False
        entry_bar = -1
        entry_price = 0.0
        position_size = 1.0

        for i in range(len(data)):
            # Record a scheduled entry only when its execution bar arrives.
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar = i
                    entry_price = opens[i]

            if entry_bar >= 0:
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_expired = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_expired:
                    # The zero decision exits at the following bar's open.
                    entry_bar = -1
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif entries[i] and np.isfinite(strengths[i]):
                position_size = float(np.clip(strengths[i], 0.25, 1.0))
                pending_entry = True
                signals[i] = 1
                sizes[i] = position_size

        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapReversalParams(hold_bars=3, profit_target=0.05)
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 101, 106, 100, 200, 201, 202, 203, 204], dtype=float)
    data = pd.DataFrame({
        "open": [100, 100, 101, 100, 200, 200, 201, 202, 203],
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1000,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, True, False, False, False, False],
        "strength": [0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0]
    assert np.allclose(result.loc[result["signal"] == 1, "size"], [0.4, 0.4, 0.8, 0.8, 0.8])
    assert (result["size"] > 0).all()
    assert not strategy.indicators(data, params)["entry"].any()
    assert strategy.warmup_bars(params) == 22


if __name__ == "__main__":
    _self_check()
