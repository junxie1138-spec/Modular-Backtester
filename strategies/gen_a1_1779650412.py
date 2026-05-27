from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapJumpParams:
    return_window: int = 20
    vol_regime_window: int = 60
    vol_regime_lookback: int = 252
    atr_window: int = 14
    z_enter: float = 1.75
    z_size_cap: float = 3.0
    base_size: float = 0.5
    atr_stop_mult: float = 2.5
    require_compressed_vol: bool = True


class GeneratedStrategy(BaseStrategy[GapJumpParams]):
    strategy_id = "gen_a1_1779650412"

    @classmethod
    def params_type(cls):
        return GapJumpParams

    @classmethod
    def warmup_bars(cls, params: GapJumpParams) -> int:
        base = params.vol_regime_window + params.vol_regime_lookback
        other = max(params.return_window, params.atr_window)
        return int(base + other + 2)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GapJumpParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        ret = close.pct_change()

        rw = int(params.return_window)
        mu = ret.rolling(rw, min_periods=rw).mean()
        sigma = ret.rolling(rw, min_periods=rw).std()
        sigma_safe = sigma.replace(0.0, np.nan)
        z = (ret - mu) / sigma_safe

        vw = int(params.vol_regime_window)
        vl = int(params.vol_regime_lookback)
        realized_vol = ret.rolling(vw, min_periods=vw).std()
        vol_median = realized_vol.rolling(vl, min_periods=vl).median()
        compressed = (realized_vol < vol_median).astype(float)
        compressed = compressed.where(vol_median.notna(), other=np.nan)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        aw = int(params.atr_window)
        atr = tr.rolling(aw, min_periods=aw).mean()

        out = pd.DataFrame(
            {
                "ret": ret,
                "z": z,
                "atr": atr,
                "compressed": compressed,
            },
            index=data.index,
        )
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapJumpParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        compressed = indicators["compressed"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        in_trade = False
        entry_size = 0.0
        high_water = -np.inf

        z_enter = float(params.z_enter)
        z_cap = float(params.z_size_cap)
        base_sz = float(params.base_size)
        stop_mult = float(params.atr_stop_mult)
        require_comp = bool(params.require_compressed_vol)

        for i in range(n):
            zi = z[i]
            ai = atr[i]
            ci = close[i]
            comp = compressed[i]

            atr_valid = np.isfinite(ai) and ai > 0.0

            if in_trade:
                if not atr_valid or not np.isfinite(ci):
                    signal[i] = 1
                    size[i] = entry_size
                    continue
                if ci > high_water:
                    high_water = ci
                stop_level = high_water - stop_mult * ai
                if ci <= stop_level:
                    in_trade = False
                    entry_size = 0.0
                    high_water = -np.inf
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size[i] = entry_size
            else:
                if not np.isfinite(zi) or not atr_valid:
                    continue
                gate_ok = (not require_comp) or (np.isfinite(comp) and comp >= 0.5)
                if gate_ok and zi >= z_enter and z_enter > 0.0:
                    mult = zi / z_enter
                    if mult > z_cap:
                        mult = z_cap
                    sz = base_sz * mult
                    if sz > 0.0 and np.isfinite(sz):
                        in_trade = True
                        entry_size = sz
                        high_water = ci
                        signal[i] = 1
                        size[i] = sz

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df.loc[df["size"] <= 0.0, "size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
