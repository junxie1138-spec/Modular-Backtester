from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    ma_window: int = 20
    season_window: int = 504
    z_entry: float = 1.2
    evidence_min: float = 1.0
    min_samples: int = 30
    hold_bars: int = 2
    size_floor: float = 0.15
    size_cap: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local-w2_1790992526"

    @classmethod
    def params_type(cls) -> type[RecoilParams]:
        return RecoilParams

    def warmup_bars(self, params: RecoilParams) -> int:
        ma = max(2, int(params.ma_window))
        season = max(1, int(params.season_window))
        hold = min(2, max(1, int(params.hold_bars)))
        return int(ma + season + hold + 1)

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        ma = max(2, int(params.ma_window))
        season = max(1, int(params.season_window))
        hold = min(2, max(1, int(params.hold_bars)))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        opening = data["open"].astype(float)
        out["ma"] = close.rolling(ma, min_periods=ma).mean()
        out["price_std"] = close.rolling(ma, min_periods=ma).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["weekday"] = pd.Series(data.index.dayofweek, index=data.index)
        # At today's close, this outcome has already exited at today's open.
        # Its decision was hold + 1 bars ago; entry was the next open.
        out["matured_return"] = (opening / opening.shift(hold).replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["matured_z"] = out["z"].shift(hold + 1)
        out["matured_weekday"] = out["weekday"].shift(hold + 1)
        out["recoil_product"] = (-out["matured_z"] * out["matured_return"]).replace([np.inf, -np.inf], np.nan)
        out["seasonal_evidence"] = np.nan
        out["seasonal_count"] = 0.0
        valid = out["recoil_product"].notna()
        for weekday in range(5):
            matching = valid & out["matured_weekday"].eq(weekday)
            product = out["recoil_product"].where(matching, 0.0)
            prefix = f"weekday_{weekday}"
            out[prefix + "_sum"] = product.rolling(season, min_periods=season).sum()
            out[prefix + "_energy"] = product.pow(2).rolling(season, min_periods=season).sum()
            out[prefix + "_count"] = matching.astype(float).rolling(season, min_periods=season).sum()
            out[prefix + "_evidence"] = (out[prefix + "_sum"] / np.sqrt(out[prefix + "_energy"].replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan)
            current = out["weekday"].eq(weekday)
            out.loc[current, "seasonal_evidence"] = out.loc[current, prefix + "_evidence"]
            out.loc[current, "seasonal_count"] = out.loc[current, prefix + "_count"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoilParams) -> SignalFrame:
        hold = min(2, max(1, int(params.hold_bars)))
        threshold = max(0.01, float(params.z_entry))
        evidence_min = max(0.01, float(params.evidence_min))
        z = indicators["z"]
        evidence = indicators["seasonal_evidence"]
        eligible = (z.notna() & evidence.notna() & z.abs().ge(threshold)
                    & evidence.ge(evidence_min)
                    & indicators["seasonal_count"].ge(max(1, int(params.min_samples))))
        eligible.iloc[:self.warmup_bars(params)] = False
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[eligible & z.lt(0)] = 1
        candidate.loc[eligible & z.gt(0)] = -1
        floor = min(1.0, max(0.01, float(params.size_floor)))
        cap = min(1.0, max(floor, float(params.size_cap)))
        displacement_strength = (z.abs() / (2.0 * threshold)).clip(0.0, 1.0)
        seasonal_strength = (evidence / (2.0 * evidence_min)).clip(0.0, 1.0)
        requested_size = (cap * displacement_strength * seasonal_strength).clip(floor, cap).fillna(floor)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), floor, dtype=float)
        candidates = candidate.to_numpy()
        requested = requested_size.to_numpy(dtype=float)
        side = 0
        exit_at = -1
        entry_size = floor
        for i in range(len(data)):
            if side != 0 and i >= exit_at:
                # An exit decision is always flat; reentry waits another bar.
                side = 0
            elif side == 0 and candidates[i] != 0:
                side = int(candidates[i])
                exit_at = i + hold
                entry_size = float(requested[i])
            signal[i] = side
            if side != 0:
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(floor).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
