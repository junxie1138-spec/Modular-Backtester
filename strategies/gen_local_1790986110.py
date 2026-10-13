from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRegimeParams:
    signature_window: int = 8
    atr_window: int = 14
    regime_window: int = 24
    persistence_on: float = 0.35
    persistence_off: float = 0.20
    area_threshold: float = 0.06
    stop_atr: float = 1.8
    hold_days: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureRegimeParams]):
    strategy_id = "gen_local_1790986110"

    @classmethod
    def params_type(cls) -> type[SignatureRegimeParams]:
        return SignatureRegimeParams

    def warmup_bars(self, params: SignatureRegimeParams) -> int:
        return int(max(2, params.atr_window) + max(2, params.signature_window, params.regime_window) + 2)

    def indicators(self, data: pd.DataFrame, params: SignatureRegimeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        length = max(2, int(params.signature_window))
        regime = max(2, int(params.regime_window))
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(max(2, int(params.atr_window))).mean()
        out["previous_atr"] = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["price_increment"] = data["close"].diff() / out["previous_atr"]
        out["range_innovation"] = out["true_range"].diff() / out["previous_atr"]
        out[["price_increment", "range_innovation"]] = out[["price_increment", "range_innovation"]].replace([np.inf, -np.inf], np.nan)
        out["price_path"] = out["price_increment"].fillna(0.0).cumsum()
        out["range_path"] = out["range_innovation"].fillna(0.0).cumsum()
        out["cross_increment"] = 0.5 * (
            out["price_path"].shift(1).fillna(0.0) * out["range_innovation"]
            - out["range_path"].shift(1).fillna(0.0) * out["price_increment"]
        )
        out["price_displacement"] = out["price_increment"].rolling(length).sum()
        out["range_displacement"] = out["range_innovation"].rolling(length).sum()
        out["signature_area"] = out["cross_increment"].rolling(length).sum() - 0.5 * (
            out["price_path"].shift(length).fillna(0.0) * out["range_displacement"]
            - out["range_path"].shift(length).fillna(0.0) * out["price_displacement"]
        )
        out["price_variation"] = out["price_increment"].abs().rolling(length).sum()
        out["range_variation"] = out["range_innovation"].abs().rolling(length).sum()
        out["area_scale"] = out["price_variation"] * out["range_variation"]
        out["normalized_area"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 1e-12)
        out["regime_displacement"] = out["price_increment"].rolling(regime).sum()
        out["regime_variation"] = out["price_increment"].abs().rolling(regime).sum()
        out["persistence"] = out["regime_displacement"].abs() / out["regime_variation"].where(out["regime_variation"] > 1e-12)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureRegimeParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        minimum = max(0.01, min(1.0, float(params.min_size)))
        maximum = max(minimum, min(1.0, float(params.max_size)))
        sizes = np.full(n, minimum, dtype=float)
        threshold = max(1e-6, float(params.area_threshold))
        holding = max(3, min(5, int(params.hold_days)))
        stop_multiple = max(0.01, float(params.stop_atr))
        on = max(float(params.persistence_on), float(params.persistence_off))
        off = min(float(params.persistence_on), float(params.persistence_off))
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        area = indicators["normalized_area"].to_numpy(dtype=float)
        persistence = indicators["persistence"].to_numpy(dtype=float)
        dx = indicators["price_increment"].to_numpy(dtype=float)
        dy = indicators["range_innovation"].to_numpy(dtype=float)
        displacement = indicators["price_displacement"].to_numpy(dtype=float)
        regime_displacement = indicators["regime_displacement"].to_numpy(dtype=float)
        ready = np.isfinite(np.column_stack([
            closing, atr, area, persistence, dx, dy, displacement, regime_displacement
        ])).all(axis=1)
        trending = False
        active = False
        pending = False
        entry_index = -1
        fixed_stop = np.nan
        position_size = minimum
        warmup = self.warmup_bars(params)

        for i in range(n):
            # A decision on the previous close fills at this bar's open.
            if pending:
                pending = False
                if i > 0 and np.isfinite(opening[i]) and opening[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry_index = i
                    fixed_stop = opening[i] - stop_multiple * atr[i - 1]

            if np.isfinite(persistence[i]):
                if persistence[i] >= on:
                    trending = True
                elif persistence[i] <= off:
                    trending = False

            if active:
                sizes[i] = position_size
                # Freeze both the entry price and ATR; this stop never trails.
                stopped = not np.isfinite(closing[i]) or closing[i] < fixed_stop
                expired = i - entry_index + 1 >= holding
                if stopped or expired:
                    active = False
                else:
                    signals[i] = 1
            elif i >= warmup - 1 and ready[i] and atr[i] > 0:
                continuation = trending and regime_displacement[i] > 0 and displacement[i] > 0 and dx[i] > 0 and area[i] > threshold
                recovery = not trending and displacement[i] < 0 and dx[i] > 0 and dy[i] < 0 and area[i] < -threshold
                if continuation or recovery:
                    conviction = min(1.0, abs(area[i]) / (3.0 * threshold))
                    position_size = minimum + (maximum - minimum) * conviction
                    signals[i] = 1
                    sizes[i] = position_size
                    pending = True

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(minimum).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
