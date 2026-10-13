from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 6
    area_threshold: float = 0.12
    stop_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791315903"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        if params.signature_window < 2 or params.hold_bars not in (1, 2):
            raise ValueError("signature_window must be >= 2; hold_bars must be 1 or 2")
        if not np.isfinite(params.area_threshold) or not 0 < params.area_threshold < 0.5:
            raise ValueError("area_threshold must be between 0 and 0.5")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        return int(max(200, 20 + params.signature_window, 15))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["volume_reference"] = volume.rolling(20, min_periods=20).mean().shift(1)
        out["dx"] = volume.div(out["volume_reference"].where(out["volume_reference"] > 0))
        out["dy"] = np.log(close).diff()
        window = params.signature_window
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric level-two signature: half the sum over ordered increment pairs.
        for lag in range(1, window):
            pairs = out["dx"].shift(lag) * out["dy"] - out["dy"].shift(lag) * out["dx"]
            area = area + 0.5 * pairs.rolling(window - lag, min_periods=window - lag).sum()
        out["signature_area"] = area
        out["volume_clock"] = out["dx"].rolling(window, min_periods=window).sum()
        out["net_return"] = out["dy"].rolling(window, min_periods=window).sum()
        out["return_variation"] = out["dy"].abs().rolling(window, min_periods=window).sum()
        denominator = out["volume_clock"] * out["return_variation"]
        out["normalized_area"] = area.div(denominator.where(denominator > 0))
        out["efficiency"] = out["net_return"].div(out["return_variation"].where(out["return_variation"] > 0))
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (close > out["ma200"])
            & (out["normalized_area"] > params.area_threshold)
            & (out["efficiency"] > 0.55)
            & (out["dy"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_atr = np.nan
        stop_price = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                if pending:
                    # The prior close's decision fills at this bar's open.
                    stop_price = opens[i] - params.stop_atr * entry_atr
                    pending = False
                held += 1
                if (not np.isfinite(stop_price) or not np.isfinite(closes[i])
                        or closes[i] < stop_price or held >= params.hold_bars):
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                pending = True
                entry_atr = atr[i]
                held = 0
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 90.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[1], "open"] = 105.0
    data.loc[index[1], "close"] = 102.0
    ind = pd.DataFrame({"entry": [True, False, False, False, True, False, False, False],
                        "atr": [2.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0, 2.0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, SignatureParams(stop_atr=1.0))
    # First trade stops after its first close using actual entry open and frozen ATR.
    # Second trade exits after two sessions; both decisions execute one bar later.
    assert result.data["signal"].tolist() == [0, 1, 0, 0, 0, 1, 1, 0]
    assert (result.data["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(SignatureParams()) >= 200
