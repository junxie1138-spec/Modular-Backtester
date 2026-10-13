from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 6
    history_bars: int = 252
    baseline_bars: int = 20
    bandwidth: float = 0.20
    volume_threshold: float = 1.20
    edge_threshold: float = 0.25
    min_matches: int = 8
    hold_bars: int = 8
    profit_target: float = 0.025
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790991398"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_bars + params.baseline_bars + params.shape_bars + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["volume_baseline"] = data["volume"].rolling(params.baseline_bars).mean().shift(1)
        out["volume_ratio"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["path_length"] = out["return"].abs().rolling(params.shape_bars).sum()
        out["orientation"] = np.sign(out["return"].rolling(params.shape_bars).sum())
        out["risk"] = out["return"].rolling(params.baseline_bars).std() * np.sqrt(params.hold_bars)
        columns = []
        for j in range(params.shape_bars):
            name = f"impulse_{j}"
            out[name] = (
                out["return"].shift(j) * out["volume_ratio"].shift(j)
                * out["orientation"] / out["path_length"].where(out["path_length"] > 0)
            )
            columns.append(name)
        # This backward outcome becomes available only when all holding bars have closed.
        entry_open = data["open"].shift(params.hold_bars - 1)
        out["matured_return"] = close / entry_open.where(entry_open > 0) - 1.0
        features = out[columns].replace([np.inf, -np.inf], np.nan)
        weighted = pd.Series(0.0, index=data.index)
        squared = pd.Series(0.0, index=data.index)
        weights = pd.Series(0.0, index=data.index)
        matches = pd.Series(0, index=data.index)
        # Disjoint analog outcomes cannot overlap the current shape window.
        for lag in range(params.hold_bars + params.shape_bars, params.history_bars + 1):
            distance = (features - features.shift(lag)).pow(2).mean(axis=1, skipna=False)
            weight = np.exp(-distance / (2.0 * params.bandwidth ** 2))
            old_risk = out["risk"].shift(lag)
            outcome = (
                out["matured_return"].shift(lag - params.hold_bars)
                * out["orientation"].shift(lag) / old_risk.where(old_risk > 0)
            ).replace([np.inf, -np.inf], np.nan)
            valid = (weight >= 0.5) & outcome.notna()
            weight = weight.where(valid, 0.0).fillna(0.0)
            outcome = outcome.fillna(0.0)
            weighted += weight * outcome
            squared += weight * outcome.pow(2)
            weights += weight
            matches += valid.astype(int)
        out["matches"] = matches
        out["weight_sum"] = weights
        out["forecast"] = weighted / weights.where(weights > 0)
        out["outcome_rms"] = np.sqrt(squared / weights.where(weights > 0))
        out["edge"] = (out["forecast"] / out["outcome_rms"].where(out["outcome_rms"] > 0)).clip(-1.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        eligible = (
            (indicators["volume_ratio"] >= params.volume_threshold)
            & (indicators["return"] * indicators["orientation"] > 0)
            & (indicators["matches"] >= params.min_matches)
            & (indicators["edge"].abs() >= params.edge_threshold)
        ).fillna(False)
        direction = np.sign(indicators["orientation"] * indicators["edge"]).fillna(0).astype(int)
        conviction = indicators["edge"].abs().fillna(0.0).clip(0.0, 1.0)
        proposed_size = params.min_size + (params.max_size - params.min_size) * conviction
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        held_size = params.min_size
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        for i in range(len(data)):
            if side != 0:
                # Entry was decided on the previous close; observe its fill only now.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                if (
                    not valid_price
                    or (np.isfinite(gain) and gain >= params.profit_target)
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    side = 0
                    entry_price = np.nan
                    # Keep this close's exit flat even if another analog qualifies.
                    continue
            elif i >= self.warmup_bars(params) and bool(eligible.iloc[i]):
                side = int(direction.iloc[i])
                entry_bar = i + 1
                entry_price = np.nan
                held_size = float(proposed_size.iloc[i])
            signals[i] = side
            if side != 0:
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
