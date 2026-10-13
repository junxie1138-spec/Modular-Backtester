from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_floor: float = 0.10
    trail_atr: float = 1.50


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791341783"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if not np.isfinite(params.area_floor) or not 0.0 <= params.area_floor < 0.5:
            raise ValueError("area_floor must be finite and in [0, 0.5)")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["range_fraction"] = ((data["high"] - data["low"]) / close).where(
            data["high"] > data["low"]
        )
        out["price_step"] = np.log(close).diff()
        out["range_step"] = np.log(out["range_fraction"]).diff()
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of the four-step price/range path.
        for earlier in range(4):
            for later in range(earlier + 1, 4):
                px = out["price_step"].shift(3 - earlier)
                rx = out["range_step"].shift(3 - earlier)
                py = out["price_step"].shift(3 - later)
                ry = out["range_step"].shift(3 - later)
                area = area + 0.5 * (px * ry - rx * py)
        out["signature_area"] = area
        out["area_scale"] = (
            out["price_step"].abs().rolling(4).sum()
            * out["range_step"].abs().rolling(4).sum()
        )
        out["normalized_area"] = area / out["area_scale"].replace(0.0, np.nan)
        out["compressed_range"] = out["range_fraction"].rolling(4).mean()
        out["reference_range"] = out["range_fraction"].shift(4).rolling(20).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14).mean()
        out["four_day_change"] = close.diff(4)
        out["daily_change"] = close.diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["normalized_area"] > params.area_floor)
            & (out["compressed_range"] < 0.8 * out["reference_range"])
            & (out["four_day_change"] > 0.0)
            & (out["daily_change"] > 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        desired = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            valid = (
                np.isfinite(prices[i]).all()
                and (prices[i] > 0.0).all()
                and high >= max(opening, close)
                and low <= min(opening, close)
                and np.isfinite(atr[i])
                and atr[i] > 0.0
            )
            if not valid:
                pending = active = armed = False
                continue
            if pending:
                pending = False
                if i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0.0:
                    continue
                # Model the next-open entry with the config's 5 bps slippage.
                entry = opening * 1.0005
                stop = entry - params.trail_atr * atr[i - 1]
                peak = entry
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                # Only the stop established before this bar is tested against its low.
                exit_now = low <= stop
                peak = max(peak, high)
                armed = armed or peak >= entry * 1.006
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                if exit_now or close <= stop or age >= 4:
                    active = False
                else:
                    desired[i] = 1
            elif entries[i]:
                desired[i] = 1
                pending = True
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.4, 101.5, 101.0, 101.0],
            "high": [100.2, 100.5, 102.0, 101.6, 101.2, 101.2],
            "low": [99.8, 99.9, 100.3, 101.0, 100.8, 100.8],
            "close": [100.0, 100.4, 101.5, 101.0, 101.0, 101.0],
            "volume": [1000.0] * 6,
        },
        index=index,
    )
    indicators = pd.DataFrame(
        {"atr": [1.0, 1.0, 0.5, 4.0, 1.0, 1.0],
         "entry": [True, False, False, False, False, False]},
        index=index,
    )
    # ATR expansion must not lower the previously raised stop.
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0.0).all()
    indicators["atr"] = 10.0
    result = strategy.generate_signals(data, indicators, None, params).data
    # Breakeven remains binding even when the ATR trail lies below entry.
    breached = data.copy()
    breached.loc[index[3], "low"] = 100.0
    result = strategy.generate_signals(breached, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    flat = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=40),
    )
    result = strategy.generate_signals(
        flat, strategy.indicators(flat, params), None, params
    ).data
    assert result["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= 24


if __name__ == "__main__":
    _self_check()
