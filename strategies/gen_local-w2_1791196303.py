from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 3
    debt_bars: int = 5
    repayment_fraction: float = 0.35
    excursion_bars: int = 20
    minimum_excursion_atr: float = 1.5
    atr_bars: int = 14
    regime_bars: int = 200
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791196303"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.regime_bars, params.excursion_bars,
                       params.atr_bars + 1,
                       params.roc_bars + params.debt_bars + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (min(params.roc_bars, params.debt_bars, params.excursion_bars,
                params.atr_bars, params.holding_bars) < 1
                or params.regime_bars != 200
                or not np.isfinite(params.repayment_fraction)
                or params.repayment_fraction <= 0
                or not np.isfinite(params.minimum_excursion_atr)
                or params.minimum_excursion_atr <= 0
                or not np.isfinite(params.stop_atr)
                or params.stop_atr <= 0):
            raise ValueError("Invalid parameters; regime_bars must equal 200")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        out["atr"] = true_range.rolling(params.atr_bars).mean()
        out["regime_ma"] = close.rolling(200).mean()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["negative_debt"] = (-out["acceleration"]).clip(lower=0).rolling(params.debt_bars).sum().shift(1)
        out["positive_debt"] = out["acceleration"].clip(lower=0).rolling(params.debt_bars).sum().shift(1)
        out["previous_acceleration"] = out["acceleration"].shift(1)
        out["recent_peak"] = close.rolling(params.excursion_bars).max()
        out["recent_floor"] = close.rolling(params.excursion_bars).min()
        safe_atr = out["atr"].where(out["atr"] > 0)
        out["drawdown_atr"] = (out["recent_peak"] - close) / safe_atr
        out["runup_atr"] = (close - out["recent_floor"]) / safe_atr
        out["ready"] = np.isfinite(out).all(axis=1)
        out["long_entry"] = (
            out["ready"]
            & (close > out["regime_ma"])
            & (out["drawdown_atr"] >= params.minimum_excursion_atr)
            & (out["roc"] < 0)
            & (out["previous_acceleration"] <= 0)
            & (out["acceleration"] > 0)
            & (out["negative_debt"] > 0)
            & (out["acceleration"] >= params.repayment_fraction * out["negative_debt"])
        ).fillna(False)
        out["short_entry"] = (
            out["ready"]
            & (close < out["regime_ma"])
            & (out["runup_atr"] >= params.minimum_excursion_atr)
            & (out["roc"] > 0)
            & (out["previous_acceleration"] >= 0)
            & (out["acceleration"] < 0)
            & (out["positive_debt"] > 0)
            & (-out["acceleration"] >= params.repayment_fraction * out["positive_debt"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry_price = 0.0
        entry_atr = 0.0
        held = 0
        for i in range(len(data)):
            if pending:
                if (i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0
                        or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0):
                    raise ValueError("Entry requires a valid opening price and prior ATR")
                position = pending
                pending = 0
                entry_price = opens[i]
                entry_atr = atr[i - 1]
                held = 0
            if position:
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    raise ValueError("An active position requires a valid closing price")
                held += 1
                stop_hit = (
                    closes[i] < entry_price - params.stop_atr * entry_atr
                    if position == 1 else
                    closes[i] > entry_price + params.stop_atr * entry_atr
                )
                if stop_hit or held >= params.holding_bars:
                    position = 0
                    continue
                raw[i] = position
            elif longs[i] or shorts[i]:
                pending = 1 if longs[i] else -1
                raw[i] = pending
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    sample = pd.DataFrame({
        "open": [100.0, 101.0, 100.0, 100.0, 101.0],
        "high": [101.0, 102.0, 101.0, 101.0, 105.0],
        "low": [99.0, 99.0, 95.0, 99.0, 100.0],
        "close": [100.0, 100.0, 96.0, 100.0, 104.0],
        "volume": [1000.0] * 5,
    }, index=index)
    features = pd.DataFrame({
        "atr": [1.0] * 5,
        "long_entry": [True, False, False, False, False],
        "short_entry": [False, False, False, True, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(sample, features, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1]
    assert (result.data["size"] > 0).all()
    features["long_entry"] = [True, False, False, False, False]
    features["short_entry"] = False
    sample["close"] = 101.0
    expiry = GeneratedStrategy.generate_signals(sample, features, None, Params(holding_bars=2))
    assert expiry.data["signal"].tolist() == [0, 1, 1, 0, 0]
