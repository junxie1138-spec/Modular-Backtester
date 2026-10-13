from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    entropy_window: int = 18
    channel_window: int = 20
    entropy_ceiling: float = 0.88
    entropy_drop: float = 0.07
    profit_target: float = 0.025
    max_hold: int = 10
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791200281"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.entropy_window + 8, params.channel_window + 2))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if (params.entropy_window < 6 or params.channel_window < 2
                or not 0.0 < params.entropy_ceiling <= 1.0
                or not 0.0 < params.entropy_drop < 1.0
                or not 0.0 < params.profit_target < 1.0
                or params.max_hold < 1
                or not np.isfinite(params.size) or params.size <= 0.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["pressure"] = ((data["close"] - data["open"]) / span).replace(
            [np.inf, -np.inf], np.nan
        )
        a, b, c = out["pressure"].shift(2), out["pressure"].shift(1), out["pressure"]
        valid = a.notna() & b.notna() & c.notna()
        # Ties retain chronological order: earlier observations rank first.
        out["pattern"] = (
            4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)
        ).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = (out["pattern"] == code).astype(float).where(valid).rolling(
                params.entropy_window, min_periods=params.entropy_window
            ).mean()
            out[f"pattern_probability_{code}"] = probability
            entropy -= probability * np.log(probability.clip(lower=1e-12))
        out["entropy"] = entropy / np.log(6.0)
        out["entropy_drop"] = out["entropy"].shift(5) - out["entropy"]
        out["upper"] = data["high"].shift(1).rolling(params.channel_window).max()
        out["lower"] = data["low"].shift(1).rolling(params.channel_window).min()
        # Both closes must exceed the channel known before the first close.
        out["long_confirm"] = (
            (data["close"].shift(1) > out["upper"].shift(1))
            & (data["close"] > out["upper"].shift(1))
        )
        out["short_confirm"] = (
            (data["close"].shift(1) < out["lower"].shift(1))
            & (data["close"] < out["lower"].shift(1))
        )
        out["complexity_gate"] = (
            (out["entropy"] <= params.entropy_ceiling)
            & (out["entropy"].shift(1) <= params.entropy_ceiling)
            & (out["entropy_drop"] >= params.entropy_drop)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        gate = indicators["complexity_gate"].fillna(False).to_numpy(dtype=bool)
        long_entry = (indicators["long_confirm"].fillna(False).to_numpy(dtype=bool) & gate)
        short_entry = (indicators["short_confirm"].fillna(False).to_numpy(dtype=bool) & gate)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position, held, entry_price = 0, 0, np.nan
        for i in range(len(data)):
            if position:
                if held == 0:
                    # Observe the actual entry-bar open only when that bar arrives.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        continue
                held += 1
                gain = position * (closes[i] / entry_price - 1.0)
                # Close-observed exits execute at the following open.
                if gain >= params.profit_target or held >= params.max_hold:
                    position = 0
                    held = 0
                raw[i] = position
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                held = 0
                raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[2], "close"] = 103.0
    data.loc[index[6], "close"] = 97.0
    ind = pd.DataFrame(
        {"complexity_gate": True, "long_confirm": False, "short_confirm": False},
        index=index,
    )
    ind.loc[index[0], "long_confirm"] = True
    ind.loc[index[4], "short_confirm"] = True
    ind.loc[index[8], "long_confirm"] = True
    params = GeneratedParams(max_hold=2)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 0]
    # Separate target exits from time stops.
    params.max_hold = 10
    targeted = strategy.generate_signals(data, ind, ctx, params).data
    assert targeted["signal"].iloc[3] == 0 and targeted["signal"].iloc[7] == 0
    assert targeted["signal"].iloc[11] == 1
    # A first entry's overnight gap must alter the target reference.
    data.loc[index[1], "open"] = 104.0
    gapped = strategy.generate_signals(data, ind, ctx, params).data
    assert gapped["signal"].iloc[3] == 1
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:8], strategy.indicators(data.iloc[:8], params))
    assert (result["size"] > 0.0).all()
    data.loc[index[9], ["close", "high"]] = [106.0, 107.0]
    data.loc[index[10], ["close", "high"]] = [106.5, 108.0]
    confirmed = strategy.indicators(data, GeneratedParams(channel_window=2))
    assert not confirmed["long_confirm"].iloc[9]
    assert confirmed["long_confirm"].iloc[10]
    long_index = pd.date_range("2021-01-01", periods=90, freq="D")
    t = np.arange(90, dtype=float)
    close = 100.0 + t * 0.1 + np.sin(t)
    history = pd.DataFrame(
        {"open": close - np.sin(t * 1.7), "high": close + 2.0,
         "low": close - 2.0, "close": close, "volume": 1000.0}, index=long_index
    )
    full = strategy.indicators(history, params)
    assert full["entropy"].iloc[60:].notna().all()
    pd.testing.assert_frame_equal(
        full.iloc[:60], strategy.indicators(history.iloc[:60], params)
    )


if __name__ == "__main__":
    _self_check()
