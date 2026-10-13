from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 12
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790931402"

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(15, params.lookback + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.lookback < 3 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("lookback must be >= 3 and stop_atr must be positive and finite")
        w = params.lookback
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        x = out["return"].fillna(0.0)
        y = out["lag_return"].fillna(0.0)
        out["prefix_x"] = x.cumsum().shift(1, fill_value=0.0)
        out["prefix_y"] = y.cumsum().shift(1, fill_value=0.0)
        out["cross_increment"] = out["prefix_x"] * y - out["prefix_y"] * x
        out["sum_x"] = x.rolling(w, min_periods=w).sum()
        out["sum_y"] = y.rolling(w, min_periods=w).sum()
        out["signature_area"] = 0.5 * (
            out["cross_increment"].rolling(w, min_periods=w).sum()
            - out["prefix_x"].shift(w - 1) * out["sum_y"]
            + out["prefix_y"].shift(w - 1) * out["sum_x"]
        )
        out["area_scale"] = (
            x.abs().rolling(w, min_periods=w).sum()
            * y.abs().rolling(w, min_periods=w).sum()
        )
        out["area"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["autocorrelation"] = out["return"].rolling(w, min_periods=w).corr(out["lag_return"])
        out["drift"] = out["return"].rolling(w, min_periods=w).mean()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        valid_pair = out[["return", "lag_return"]].notna().all(axis=1)
        out["ready"] = (
            valid_pair.rolling(w, min_periods=w).sum().eq(w)
            & out[["area", "autocorrelation", "drift", "atr"]].notna().all(axis=1)
            & out["atr"].gt(0)
        )
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        area = indicators["area"]
        entry_event = (
            indicators["ready"].fillna(False)
            & area.gt(0.04)
            & area.shift(1).le(0.04)
            & indicators["autocorrelation"].gt(0.15)
        ).fillna(False)
        candidates = np.where(entry_event, np.sign(indicators["drift"].fillna(0.0)), 0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # The previous close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        signals[i] = 0
                        continue
                    stop = opens[i] - position * params.stop_atr * entry_atr
                # Freeze the entry ATR and stop; mirror the downside stop for shorts.
                stopped = not np.isfinite(closes[i]) or position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= 8
                if stopped or expired:
                    position = 0
            elif (
                candidates[i] != 0
                and np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
            ):
                position = int(candidates[i])
                entry_bar = i + 1
                entry_atr = atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check both stop directions, next-open anchoring, and exactly one bar of delay.
    index = pd.date_range("2020-01-01", periods=5)
    for direction in (1, -1):
        data = pd.DataFrame({
            "open": [100.0, 100.0, 100.0 + 10.0 * direction, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0 + 8.0 * direction, 100.0 + 5.0 * direction, 100.0],
        }, index=index)
        indicators = pd.DataFrame({
            "area": [0.0, 0.2, 0.2, 0.2, 0.2],
            "autocorrelation": 0.5,
            "drift": float(direction),
            "atr": 2.0,
            "ready": True,
        }, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None, SignatureParams()).data
        assert result["signal"].tolist() == [0, 0, direction, direction, 0]
        assert result["size"].gt(0).all()
