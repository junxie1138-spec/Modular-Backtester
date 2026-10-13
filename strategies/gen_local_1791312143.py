from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 20
    energy_threshold: float = 1.5
    breakout_window: int = 5
    profit_target: float = 0.008
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791312143"

    @classmethod
    def params_type(cls):
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return int(max(200, params.breakout_window + 1, params.energy_window + 11))

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.breakout_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not np.isfinite(params.energy_threshold) or params.energy_threshold <= 1:
            raise ValueError("Energy threshold must exceed one.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Profit target must be positive.")
        if params.hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["return"] = out["log_close"].diff()
        ratios = []
        for width in (1, 2, 4):
            mean = out["return"].rolling(width, min_periods=width).mean()
            out[f"haar_{width}"] = (mean - mean.shift(width)) * np.sqrt(width / 2.0)
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2).rolling(3, min_periods=3).mean()
            out[f"baseline_{width}"] = out[f"energy_{width}"].shift(1).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            column = f"ratio_{width}"
            out[column] = out[f"energy_{width}"] / out[f"baseline_{width}"].where(
                out[f"baseline_{width}"] > 0
            )
            ratios.append(column)
        out["energy_floor"] = out[ratios].min(axis=1, skipna=False)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["upper"] = data["high"].shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        out["lower"] = data["low"].shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).min()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        columns = ["energy_floor", "ma200", "upper", "lower"]
        ready = indicators[columns].notna().all(axis=1)
        burst = ready & (indicators["energy_floor"] >= params.energy_threshold)
        long_entry = burst & (data["close"] > indicators["upper"]) & (
            data["close"] > indicators["ma200"]
        )
        short_entry = burst & (data["close"] < indicators["lower"]) & (
            data["close"] < indicators["ma200"]
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid else np.nan
                # Close-confirmed targets exit at the following open.
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    side = 0
            elif candidates[i] and i + 1 < len(data):
                side = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100., 100., 100., 100., 100., 100., 100.],
        "high": [102., 102., 101., 100., 100., 100., 100.],
        "low": [99., 99., 99., 98., 98., 98., 99.],
        "close": [101., 101., 100., 99., 99.5, 99.5, 100.],
        "volume": 1000.,
    }, index=index)
    indicators = pd.DataFrame({
        "energy_floor": [2., 0., 0., 2., 0., 0., 0.],
        "ma200": 100., "upper": 100., "lower": 100.,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200
    assert GeneratedStrategy.indicators(data, params)["ma200"].isna().all()


if __name__ == "__main__":
    _self_check()
