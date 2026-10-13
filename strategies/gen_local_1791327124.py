from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    shock_window: int = 20
    atr_multiple: float = 3.0


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = "gen_local_1791327124"

    @classmethod
    def params_type(cls) -> type[ShockParams]:
        return ShockParams

    def warmup_bars(self, params: ShockParams) -> int:
        return int(max(params.shock_window + 2, 15))

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        if not 8 <= params.shock_window <= 252:
            raise ValueError("shock_window must be between 8 and 252")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")
        out = pd.DataFrame(index=data.index)
        window = params.shock_window
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_downside"] = out["return"].shift(1).clip(upper=0.0)
        out["shock_cross_product"] = out["lagged_downside"] * out["return"]
        out["shock_square"] = out["lagged_downside"].pow(2)
        out["shock_cross_sum"] = out["shock_cross_product"].rolling(window).sum()
        out["shock_square_sum"] = out["shock_square"].rolling(window).sum()
        out["transmission"] = out["shock_cross_sum"] / out["shock_square_sum"].replace(0.0, np.nan)
        out["negative_shock"] = out["lagged_downside"].lt(0).astype(float).where(out["lagged_downside"].notna())
        out["shock_count"] = out["negative_shock"].rolling(window).sum()
        out["return_sum"] = out["return"].rolling(window).sum()
        out["return_square_sum"] = out["return"].pow(2).rolling(window).sum()
        out["drift_strength"] = out["return_sum"] / np.sqrt(out["return_square_sum"].replace(0.0, np.nan))
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14).mean()
        out["qualified"] = (
            out["transmission"].lt(-0.15)
            & out["drift_strength"].gt(0.8)
            & out["shock_count"].ge(4)
            & out["atr"].gt(0)
        )
        out["entry"] = out["qualified"] & ~out["qualified"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry_price = 0.0
        peak = 0.0
        stop = 0.0
        entry_atr = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Match the supplied configuration's five-basis-point buy slippage.
                    entry_price = opens[i] * 1.0005
                    peak = entry_price
                    stop = entry_price - params.atr_multiple * entry_atr
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                breached = (np.isfinite(lows[i]) and lows[i] <= stop) or (valid_close and closes[i] <= stop)
                if breached or age >= 18:
                    active = False
                    continue
                if valid_close:
                    peak = max(peak, closes[i])
                    if closes[i] >= entry_price * 1.02:
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak - params.atr_multiple * atrs[i])
                # New stop levels take effect after this close; exits execute next open.
                signals[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                entry_atr = atrs[i]
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    close = pd.Series([100, 100, 103, 104, 103, 101, 99, 99], index=index, dtype=float)
    opening = close.shift(1).fillna(100)
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + 0.25, "low": np.minimum(opening, close) - 0.25, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "entry": [True, False, False, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, ShockParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()
