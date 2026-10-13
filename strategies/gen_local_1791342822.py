from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 6
    area_threshold: float = 0.20
    pressure_window: int = 4
    pressure_threshold: float = 0.15
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791342822"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.area_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.area_window < 2 or params.pressure_window < 1:
            raise ValueError("Invalid indicator windows")
        if not (0 < params.area_threshold < 1 and 0 < params.pressure_threshold < 1):
            raise ValueError("Thresholds must lie between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["absolute_return"] = out["log_return"].abs()
        length = params.area_window
        # Antisymmetric level-two signature of (time, log price).
        # Weight k is 2*k-length+1, from oldest to newest increment.
        area = pd.Series(0.0, index=data.index)
        for lag in range(length):
            area += 0.5 * (length - 1 - 2 * lag) * out["log_return"].shift(lag)
        out["signature_area"] = area
        out["area_scale"] = 0.5 * (length - 1) * out["absolute_return"].rolling(length).sum()
        out["area_score"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["close_location"] = ((2 * data["close"] - data["high"] - data["low"]) / out["range"].where(out["range"] > 0)).clip(-1, 1)
        out.loc[out["range"].eq(0), "close_location"] = 0.0
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["flow_sum"] = out["signed_volume"].rolling(params.pressure_window).sum()
        out["volume_sum"] = out["valid_volume"].rolling(params.pressure_window).sum()
        out["pressure"] = out["flow_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["ready"] = out["area_score"].notna() & out["pressure"].notna()
        out["area_score"] = out["area_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["pressure"] = out["pressure"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["candidate"] = np.select(
            [
                out["ready"] & (out["area_score"] > params.area_threshold) & (out["pressure"] > params.pressure_threshold),
                out["ready"] & (out["area_score"] < -params.area_threshold) & (out["pressure"] < -params.pressure_threshold),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        if params.max_hold_bars not in (1, 2) or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Use a positive target and a one- or two-session holding limit")
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and np.isfinite(closes[i]) else np.nan
                # ponytail: target uses observed close versus entry open; use broker fill state if exact net-fill targets are needed.
                if not valid_entry or gain >= params.profit_target or age >= params.max_hold_bars:
                    position = 0
                    continue
                signals[i] = position
            elif i >= warmup - 1 and candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SignatureParams(area_window=2, pressure_window=1)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for direction in (1, -1):
        candidate = pd.DataFrame({"candidate": [0, 0, direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, candidate, ctx, params).data
        assert result["signal"].tolist() == [0, 0, 0, direction, direction, 0, 0, 0]
        target_data = data.copy()
        target_data.loc[index[3], "close"] = 100.0 * (1 + direction * 0.01)
        result = strategy.generate_signals(target_data, candidate, ctx, params).data
        assert result["signal"].tolist() == [0, 0, 0, direction, 0, 0, 0, 0]
    flat = strategy.indicators(data, params)
    assert flat["candidate"].eq(0).all()
    assert strategy.generate_signals(data, flat, ctx, params).data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
