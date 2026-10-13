from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnBreakoutParams:
    return_window: int = 6
    breakout_window: int = 42
    atr_window: int = 14
    minimum_snr: float = 0.8
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[ReturnBreakoutParams]):
    strategy_id = "gen_local_1791252210"

    @classmethod
    def params_type(cls) -> type[ReturnBreakoutParams]:
        return ReturnBreakoutParams

    def warmup_bars(self, params: ReturnBreakoutParams) -> int:
        return int(max(params.return_window + params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ReturnBreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        returns = out["log_return"]
        window = params.return_window
        out["return_sum"] = returns.rolling(window, min_periods=window).sum()
        out["largest_positive"] = returns.clip(lower=0).rolling(window, min_periods=window).max()
        out["return_energy"] = returns.pow(2).rolling(window, min_periods=window).sum()
        out["trimmed_return"] = out["return_sum"] - out["largest_positive"]
        out["residual_noise"] = np.sqrt((out["return_energy"] - out["largest_positive"].pow(2)).clip(lower=0))
        out["snr"] = (out["trimmed_return"] / out["residual_noise"].where(out["residual_noise"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["breakout_level"] = out["trimmed_return"].shift(1).rolling(params.breakout_window, min_periods=params.breakout_window).max()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["entry"] = (
            (out["trimmed_return"] > out["breakout_level"])
            & (out["trimmed_return"].shift(1) <= out["breakout_level"].shift(1))
            & (out["trimmed_return"] > 0)
            & (returns > 0)
            & (out["snr"] >= params.minimum_snr)
            & (out["atr"] > 0)
        ).fillna(False)
        out["conviction"] = (out["snr"] / np.sqrt(float(window))).clip(lower=0, upper=1).fillna(0)
        out["entry_size"] = 0.25 + 0.70 * out["conviction"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnBreakoutParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        active = False
        pending = False
        age = 0
        stop = np.nan
        entry_atr = np.nan
        position_size = 0.25
        for i in range(count):
            if active:
                if pending:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        pending = False
                        continue
                    # Freeze the stop using the next-bar entry open and decision-bar ATR.
                    stop = opens[i] - params.stop_atr * entry_atr
                    pending = False
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= params.holding_bars:
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(entry_sizes[i]):
                active = True
                pending = True
                age = 0
                entry_atr = atrs[i]
                position_size = float(np.clip(entry_sizes[i], 0.25, 0.95))
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [90.0, 100.0, 110.0, 97.0, 95.0, 95.0],
        "high": [91.0, 111.0, 111.0, 98.0, 96.0, 96.0],
        "low": [89.0, 99.0, 96.0, 94.0, 94.0, 94.0],
        "close": [90.0, 110.0, 97.0, 95.0, 95.0, 95.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": [2.0] * 6,
        "entry_size": [0.70] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, ReturnBreakoutParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[index[1:4], "size"], 0.70)
    assert (result["size"] > 0).all()
    timed = GeneratedStrategy.generate_signals(None, data, indicators, None, ReturnBreakoutParams(holding_bars=2)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
