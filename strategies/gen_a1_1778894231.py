from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalZAnomalyParams:
    ma_len: int = 50
    z_len: int = 60
    entry_z: float = 1.2
    exit_z: float = 0.8
    snr_len: int = 10
    snr_k: float = 0.5
    regime_ma: int = 200
    use_regime: bool = True
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[SeasonalZAnomalyParams]):
    strategy_id = "gen_a1_1778894231"

    @classmethod
    def params_type(cls) -> type[SeasonalZAnomalyParams]:
        return SeasonalZAnomalyParams

    @staticmethod
    def warmup_bars(params: SeasonalZAnomalyParams) -> int:
        base = max(params.ma_len + params.z_len, params.regime_ma)
        return int(base + params.snr_len + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalZAnomalyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ind = pd.DataFrame(index=data.index)

        # distance-from-MA, expressed as a fractional gap
        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
        dist = close / ma - 1.0

        # rolling z-score of the distance
        d_mean = dist.rolling(params.z_len, min_periods=params.z_len).mean()
        d_std = dist.rolling(params.z_len, min_periods=params.z_len).std()
        d_std = d_std.where(d_std > 0.0)
        z = (dist - d_mean) / d_std
        z = z.replace([np.inf, -np.inf], np.nan)

        # trailing month-of-year climatology of z (expanding, current bar excluded)
        month = pd.Series(data.index.month, index=data.index)
        tmp = pd.DataFrame({"z": z, "month": month})
        clim = tmp.groupby("month")["z"].transform(
            lambda s: s.expanding(min_periods=1).mean().shift(1)
        )
        clim = clim.fillna(0.0)

        # de-seasonalized z-score anomaly
        z_anom = z - clim

        # signal-to-noise reference: jitter of the z-score
        noise = z.diff().rolling(params.snr_len, min_periods=params.snr_len).std()

        regime_ma = close.rolling(params.regime_ma, min_periods=params.regime_ma).mean()

        ind["z"] = z
        ind["z_clim"] = clim
        ind["z_anom"] = z_anom
        ind["noise"] = noise
        ind["regime_ma"] = regime_ma
        ind["close"] = close
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalZAnomalyParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        z_anom = indicators["z_anom"].to_numpy(dtype=float)
        noise = indicators["noise"].to_numpy(dtype=float)
        regime_ma = indicators["regime_ma"].to_numpy(dtype=float)
        close = indicators["close"].to_numpy(dtype=float)

        # raw entry: de-seasonalized z deep in discount territory
        entry_raw = z_anom < -params.entry_z
        entry_raw = np.where(np.isnan(z_anom), False, entry_raw)

        # two-bar confirmation: discount condition held this bar AND the prior bar
        entry_prev = np.concatenate(([False], entry_raw[:-1]))
        entry_conf = entry_raw & entry_prev

        # signal-to-noise gate: anomaly must stand above recent z jitter
        with np.errstate(invalid="ignore"):
            snr_ok = np.abs(z_anom) > (params.snr_k * noise)
        snr_ok = np.where(np.isnan(noise) | np.isnan(z_anom), False, snr_ok)

        # mirror exit: entry condition flips to a seasonal premium
        exit_cond = z_anom > params.exit_z
        exit_cond = np.where(np.isnan(z_anom), False, exit_cond)

        if params.use_regime:
            regime_ok = close > regime_ma
            regime_ok = np.where(np.isnan(regime_ma), False, regime_ok)
        else:
            regime_ok = np.ones(n, dtype=bool)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.base_size, dtype=float)

        in_pos = False
        held_size = params.base_size
        for i in range(n):
            if not in_pos:
                if bool(entry_conf[i]) and bool(snr_ok[i]) and bool(regime_ok[i]):
                    in_pos = True
                    if params.entry_z > 0.0:
                        conv = abs(z_anom[i]) / params.entry_z
                    else:
                        conv = 1.0
                    if not np.isfinite(conv):
                        conv = 1.0
                    held_size = float(np.clip(conv, 0.5, 1.5)) * params.base_size
                    signal[i] = 1
                    size[i] = held_size
            else:
                if bool(exit_cond[i]):
                    in_pos = False
                    signal[i] = 0
                    size[i] = params.base_size
                else:
                    signal[i] = 1
                    size[i] = held_size

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        safe_size = np.where(np.isfinite(size) & (size > 0.0), size, params.base_size)
        df["size"] = safe_size.astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
