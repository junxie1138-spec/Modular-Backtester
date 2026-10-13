from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PhaseParams:
    window: int = 24
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[PhaseParams]):
    strategy_id = "gen_local-w2_1791339972"

    @classmethod
    def params_type(cls) -> type[PhaseParams]:
        return PhaseParams

    @staticmethod
    def warmup_bars(params: PhaseParams) -> int:
        if params.window < 6:
            raise ValueError("window must be at least 6")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        return int(max(params.window + 3, 15))

    def indicators(self, data: pd.DataFrame, params: PhaseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag1_return"] = out["return"].shift(1)
        out["lag2_return"] = out["return"].shift(2)
        rolling = out["return"].rolling(params.window, min_periods=params.window)
        out["corr1"] = rolling.corr(out["lag1_return"]).replace([np.inf, -np.inf], np.nan)
        out["corr2"] = rolling.corr(out["lag2_return"]).replace([np.inf, -np.inf], np.nan)
        out["rms"] = out["return"].pow(2).rolling(params.window, min_periods=params.window).mean().pow(0.5)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out.loc[previous_close.isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["phase_forecast"] = out["corr1"] * out["return"]
        ready = out[["return", "corr1", "corr2", "rms", "atr"]].notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["corr2"] > 0.10)
            & (out["corr1"].abs() > 0.15)
            & (out["phase_forecast"] > 0)
            & (out["return"].abs() > 0.35 * out["rms"])
            & (out["rms"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PhaseParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        stop = np.nan

        for i in range(len(data)):
            # Previous close's decision fills at this bar's open.
            if not active and i > 0 and raw[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    age = 0
                    # Freeze both entry price and decision-time ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                else:
                    continue

            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= 2:
                    active = False
                    raw[i] = 0
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                raw[i] = 1

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = PhaseParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [101.0] * 8,
        "low": [96.0] * 8,
        "close": [100.0, 97.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    features = pd.DataFrame({
        "atr": [1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "entry": [True, False, False, True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    # First trade stops after one session despite ATR expansion; second lasts two.
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    warmup = strategy.indicators(data, params)
    assert not warmup["entry"].any()
    assert strategy.warmup_bars(params) >= params.window + 3


if __name__ == "__main__":
    _self_check()
