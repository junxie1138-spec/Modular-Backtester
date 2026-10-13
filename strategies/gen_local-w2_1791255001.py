from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_bars: int = 20
    shape_bars: int = 4
    history_bars: int = 252
    neighbors: int = 9
    z_entry: float = 0.75
    min_edge: float = 0.001
    min_win_share: float = 0.6
    atr_bars: int = 14
    stop_atr: float = 1.5
    holding_days: int = 2
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791255001"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(
            params.ma_bars + params.history_bars + params.holding_days + 2 * params.shape_bars,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (
            params.ma_bars >= 2 and params.shape_bars >= 2
            and params.history_bars >= params.neighbors >= 1
            and params.atr_bars >= 1 and params.holding_days in (1, 2)
            and params.z_entry > 0 and params.min_edge >= 0
            and 0.5 <= params.min_win_share <= 1
            and params.stop_atr > 0 and params.entry_slippage_bps >= 0
        ):
            raise ValueError("Invalid analog strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_bars).mean()
        out["sd"] = close.rolling(params.ma_bars).std(ddof=0).replace(0, np.nan)
        out["z"] = ((close - out["ma"]) / out["sd"]).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_bars).mean()
        out["realized_return"] = (
            data["open"] / data["open"].shift(params.holding_days).replace(0, np.nan) - 1
        ).replace([np.inf, -np.inf], np.nan)
        for lag in range(params.shape_bars):
            out[f"shape_{lag}"] = out["z"].shift(lag)
            # At maturity m, the analog decision was m - holding_days - 1.
            out[f"mature_shape_{lag}"] = out["z"].shift(params.holding_days + 1 + lag)
        current = out[[f"shape_{j}" for j in range(params.shape_bars)]].to_numpy()
        matured = out[[f"mature_shape_{j}" for j in range(params.shape_bars)]].to_numpy()
        outcomes = out["realized_return"].to_numpy()
        edge = np.full(len(data), np.nan)
        win_share = np.full(len(data), np.nan)
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(current[i]).all():
                continue
            # Exclude recent analog outcomes overlapping the current shape.
            end = i - params.shape_bars + 1
            start = max(0, end - params.history_bars)
            candidates = matured[start:end]
            rewards = outcomes[start:end]
            valid = np.isfinite(candidates).all(axis=1) & np.isfinite(rewards)
            if valid.sum() < params.neighbors:
                continue
            distance = np.mean((candidates[valid] - current[i]) ** 2, axis=1)
            nearest = np.argsort(distance, kind="stable")[:params.neighbors]
            selected = rewards[valid][nearest]
            edge[i] = selected.mean()
            win_share[i] = (selected > 0).mean()
        out["analog_edge"] = edge
        out["analog_win_share"] = win_share
        out["z_agrees"] = (
            (out["z"] < -params.z_entry) & (out["z"] > out["z"].shift(1))
        )
        out["analog_agrees"] = (
            (out["analog_edge"] > params.min_edge)
            & (out["analog_win_share"] >= params.min_win_share)
        )
        out["entry"] = (
            out["z_agrees"] & out["analog_agrees"]
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        frozen_atr = np.nan
        stop_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Capture the fill only when its open is observable.
                    entry_price = opens[i] * (1 + params.entry_slippage_bps / 10000)
                    stop_price = entry_price - params.stop_atr * frozen_atr
                age = i - entry_bar + 1
                if (
                    not np.isfinite(closes[i]) or not np.isfinite(stop_price)
                    or closes[i] < stop_price or age >= params.holding_days
                ):
                    entry_bar = -1
                    continue
                signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                frozen_atr = atr[i]
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 97.0,
        "close": [100.0, 98.0, 100.0, 100.0, 99.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, True, False, False, False],
        "atr": [1.0, 99.0, 1.0, 1.0, 0.1, 1.0, 1.0],
    }, index=index)
    params = AnalogParams(entry_slippage_bps=0.0)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
