from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapResponseParams:
    regime_window: int = 30
    atr_window: int = 14
    response_threshold: float = 0.25
    gap_atr: float = 0.35
    breakeven_pct: float = 0.01
    initial_stop_atr: float = 2.0
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[GapResponseParams]):
    strategy_id = "gen_local-w2_1790967045"

    @classmethod
    def params_type(cls):
        return GapResponseParams

    def warmup_bars(self, params: GapResponseParams) -> int:
        return params.regime_window + params.atr_window + 3

    def indicators(self, data: pd.DataFrame, params: GapResponseParams) -> pd.DataFrame:
        if params.regime_window < 2 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; regime_window must be >= 2.")
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between 3 and 5.")
        values = (params.response_threshold, params.gap_atr, params.breakeven_pct,
                  params.initial_stop_atr, params.trail_atr)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Thresholds and stop distances must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda s: s > 0)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        scale = out["atr"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] - previous) / scale
        out["body"] = (data["close"] - data["open"]) / scale
        out["gap_energy"] = out["gap"].pow(2).rolling(params.regime_window).sum()
        out["response_energy"] = (out["gap"] * out["body"]).rolling(params.regime_window).sum()
        out["response"] = (
            out["response_energy"] / out["gap_energy"].where(lambda s: s > 1e-12)
        ).shift(1)
        out["recovery"] = (data["close"] - data["open"]) / (
            previous - data["open"]
        ).where(lambda s: s > 0)
        out["elastic"] = out["response"] < -params.response_threshold
        out["plastic"] = out["response"] > params.response_threshold
        out["setup"] = (
            (out["elastic"] & (out["gap"] < -params.gap_atr) & (out["recovery"] >= 0.5))
            | (out["plastic"] & (out["gap"] > params.gap_atr) & (out["body"] > 0))
        ) & (data["close"] > data["open"])
        out["confirmation"] = (
            (data["close"] > data["close"].shift(1))
            & (data["close"] > data["open"])
        )
        out["entry"] = out["setup"].shift(1, fill_value=False) & out["confirmation"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapResponseParams) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry_price = peak = stop = 0.0
        entry_bar = 0
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite(prices[i]).all() and (prices[i] > 0).all()
            prior_atr = atr[i - 1] if i > 0 else np.nan
            if pending:
                pending = False
                if valid and np.isfinite(prior_atr) and prior_atr > 0:
                    # The prior close's entry decision fills at this open.
                    active = True
                    armed = False
                    entry_price = peak = opening
                    entry_bar = i
                    stop = entry_price - params.initial_stop_atr * prior_atr
            if active:
                if not valid or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                # Test the previously established stop before using today's high.
                if low <= stop:
                    active = False
                    continue
                peak = max(peak, high)
                if peak >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                # A newly raised stop is checked against the close, not an earlier low.
                if close <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
                    continue
                raw[i] = 1
            elif valid and entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                raw[i] = 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 100.5, 101.3, 101.0, 100.0, 100.0, 100.0],
        "high": [100.5, 100.5, 102.0, 101.7, 101.2, 100.5, 100.5, 100.5],
        "low": [99.0, 99.0, 100.0, 100.8, 100.4, 99.0, 99.0, 99.0],
        "close": [100.0, 100.0, 101.5, 101.2, 100.8, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
        "entry": [True, False, False, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapResponseParams(max_hold=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    # ATR expansion cannot lower the 100.5 stop; the breach exits next open.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    prefix = strategy.generate_signals(data.iloc[:4], indicators.iloc[:4], ctx, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:4])
    tape = pd.DataFrame({
        "open": [100.0, 99.0, 99.2, 99.4, 99.6, 99.8, 100.0, 100.2],
        "close": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4],
        "volume": [1000.0] * 8,
    }, index=index)
    tape["high"] = tape["close"] + 0.5
    tape["low"] = tape["open"] - 0.5
    derived = strategy.indicators(tape, GapResponseParams(regime_window=2, atr_window=1))
    assert derived["setup"].iloc[3] and not derived["entry"].iloc[3]
    assert derived["entry"].iloc[4]
    pd.testing.assert_frame_equal(
        strategy.indicators(tape.iloc[:5], GapResponseParams(regime_window=2, atr_window=1)),
        derived.iloc[:5],
    )


if __name__ == "__main__":
    _self_check()
