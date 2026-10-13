from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_jump: float = 1.2
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791324287"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return 28

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        scale = (atr.shift(1) / previous).where(lambda s: s > 0)
        x = np.log(close).diff() / scale
        y = np.log(volume).diff() / 0.25
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, 6):
            wedge = x.shift(lag) * y - y.shift(lag) * x
            area = area + 0.5 * wedge.rolling(6 - lag, min_periods=6 - lag).sum()
        prior_area = area.shift(6)
        drift = x.rolling(6, min_periods=6).sum()
        participation = y.rolling(6, min_periods=6).sum()
        jump = area - prior_area
        valid = (
            np.isfinite(area) & np.isfinite(prior_area)
            & np.isfinite(drift) & np.isfinite(participation)
            & np.isfinite(atr) & (atr > 0)
        )
        phase_flip = (area * prior_area < 0) & (jump.abs() >= params.area_jump)
        long_entry = valid & phase_flip & (area < 0) & (drift > 0) & (participation > 0)
        short_entry = valid & phase_flip & (area > 0) & (drift < 0) & (participation > 0)
        event = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index)
        return pd.DataFrame({
            "atr": atr,
            "price_increment": x,
            "volume_increment": y,
            "signature_area": area,
            "prior_area": prior_area,
            "area_jump": jump,
            "price_drift": drift,
            "participation": participation,
            "entry_event": event,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        if not np.isfinite(params.area_jump) or params.area_jump <= 0:
            raise ValueError("area_jump must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        events = indicators["entry_event"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        stop = np.nan
        held = 0
        for i in range(len(data)):
            if side != 0:
                if np.isnan(entry):
                    entry_atr = atr[i - 1] if i > 0 else np.nan
                    if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(entry_atr) or entry_atr <= 0:
                        side = 0
                        entry = np.nan
                        continue
                    entry = opens[i]
                    stop = entry - side * params.stop_atr * entry_atr
                    held = 0
                held += 1
                stopped = np.isfinite(closes[i]) and side * (closes[i] - stop) < 0
                if stopped or held >= 4:
                    side = 0
                    entry = np.nan
                signals[i] = side
                continue
            if events[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(events[i])
                entry = np.nan
                signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 97.0
    data.loc[index[5], "close"] = 103.0
    indicators = pd.DataFrame({"atr": 1.0, "entry_event": [1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
