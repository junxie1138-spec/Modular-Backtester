from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    window: int = 24
    atr_period: int = 14
    threshold: float = 0.12
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791312699"

    @classmethod
    def params_type(cls):
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return int(max(params.window + 9, params.atr_period + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.window < 3 or params.atr_period < 1:
            raise ValueError("Invalid indicator windows")
        if params.threshold <= 0 or params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Threshold, stop distance and holding period must be positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volatility"] = out["return"].rolling(params.window).std(ddof=0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        energy_columns = []
        forecast_columns = []
        for scale in (1, 2, 4):
            mean_column = f"mean_{scale}"
            lag_column = f"lag_return_{scale}"
            detail_column = f"haar_{scale}"
            energy_column = f"energy_{scale}"
            correlation_column = f"correlation_{scale}"
            forecast_column = f"forecast_{scale}"
            out[mean_column] = out["return"].rolling(scale).mean()
            out[f"previous_mean_{scale}"] = out[mean_column].shift(scale)
            out[detail_column] = np.sqrt(scale / 2.0) * (out[mean_column] - out[f"previous_mean_{scale}"])
            out[energy_column] = out[detail_column].pow(2).rolling(params.window).mean()
            out[lag_column] = out["return"].shift(scale)
            out[correlation_column] = out["return"].rolling(params.window).corr(out[lag_column]).clip(-1.0, 1.0)
            out[forecast_column] = out[correlation_column] * out[mean_column] / out["volatility"].replace(0.0, np.nan)
            out[f"weighted_forecast_{scale}"] = out[energy_column] * out[forecast_column]
            energy_columns.append(energy_column)
            forecast_columns.append(f"weighted_forecast_{scale}")
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=3)
        out["score"] = (out[forecast_columns].sum(axis=1, min_count=3) / out["total_energy"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["ready"] = out[energy_columns + ["atr", "volatility"]].notna().all(axis=1) & (out["atr"] > 0) & (out["volatility"] > 0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        score = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        valid = indicators["ready"].fillna(False).astype(bool)
        valid &= np.arange(len(data)) >= GeneratedStrategy.warmup_bars(params) - 1
        candidate = pd.Series(np.where(valid & (score > params.threshold), 1, np.where(valid & (score < -params.threshold), -1, 0)), index=data.index)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = confirmed.to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending = False
        entry_bar = 0
        stop_price = 0.0
        entry_atr = 0.0
        for i in range(len(data)):
            if pending:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError("Entry open must be finite and positive")
                entry_bar = i
                stop_price = opens[i] - position * params.stop_atr * entry_atr
                pending = False
            if position:
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop_price) <= 0.0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
                    signals[i] = 0
                    continue
            elif entries[i] and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0:
                position = int(entries[i])
                entry_atr = atrs[i]
                pending = True
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = WaveletParams()
    warmup = GeneratedStrategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=warmup + 25, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    for direction in (1, -1):
        indicators = pd.DataFrame({"score": float(direction), "ready": True, "atr": 1.0}, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].iloc[:warmup + 1].eq(0).all()
        assert result["signal"].iloc[warmup + 1] == direction
        assert result["signal"].iloc[warmup + params.hold_bars] == direction
        assert result["signal"].iloc[warmup + params.hold_bars + 1] == 0
        shocked = data.copy()
        shocked.iloc[warmup + 2, shocked.columns.get_loc("close")] = 100.0 - direction * 10.0
        result = GeneratedStrategy.generate_signals(shocked, indicators, None, params).data
        assert result["signal"].iloc[warmup + 2] == direction
        assert result["signal"].iloc[warmup + 3] == 0
        assert result["size"].gt(0).all()
    calculated = GeneratedStrategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert calculated["score"].eq(0.0).all()


if __name__ == "__main__":
    _self_check()
