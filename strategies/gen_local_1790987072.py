from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history: int = 252
    neighbors: int = 12
    min_edge: float = 0.0015
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    atr_period: int = 14


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790987072"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.history + 33, params.atr_period + 1, 33))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        returns = np.log(close).diff()
        r = [returns.shift(j) for j in range(8)]
        fine = sum((r[j] - r[j + 1]) ** 2 / 2.0 for j in (0, 2, 4, 6))
        middle = sum((r[j] + r[j + 1] - r[j + 2] - r[j + 3]) ** 2 / 4.0 for j in (0, 4))
        coarse = (sum(r[:4]) - sum(r[4:])) ** 2 / 8.0
        power = returns.pow(2).rolling(8, min_periods=8).sum().replace(0.0, np.nan)
        baseline = returns.pow(2).rolling(32, min_periods=32).mean().replace(0.0, np.nan)
        out["fine_energy"] = fine / power
        out["middle_energy"] = middle / power
        out["coarse_energy"] = coarse / power
        out["drift"] = sum(r) / np.sqrt(8.0 * power)
        out["rebound"] = (r[0] + r[1]) / np.sqrt(2.0 * power)
        out["relative_power"] = 0.25 * np.log((power / (8.0 * baseline)).clip(0.05, 20.0))
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        features = out[[
            "fine_energy", "middle_energy", "coarse_energy",
            "drift", "rebound", "relative_power",
        ]].to_numpy(dtype=float)
        valid = np.isfinite(features).all(axis=1)
        opens = data["open"].to_numpy(dtype=float)
        edge = np.zeros(len(data), dtype=float)
        wins = np.zeros(len(data), dtype=float)
        dispersion = np.zeros(len(data), dtype=float)
        similarity = np.zeros(len(data), dtype=float)
        count = np.zeros(len(data), dtype=int)
        k = max(3, int(params.neighbors))
        # Compare only nonoverlapping, fully matured analogs.
        for i in range(self.warmup_bars(params), len(data)):
            if not valid[i]:
                continue
            candidates = np.arange(max(32, i - params.history), i - 7)
            candidates = candidates[(i - candidates) % 3 == 0]
            candidates = candidates[valid[candidates]]
            if len(candidates) < k:
                continue
            entry_open = opens[candidates + 1]
            exit_open = opens[candidates + 3]
            usable = np.isfinite(entry_open) & np.isfinite(exit_open) & (entry_open > 0) & (exit_open > 0)
            candidates = candidates[usable]
            if len(candidates) < k:
                continue
            distance = np.square(features[candidates] - features[i]).mean(axis=1)
            chosen = np.argpartition(distance, k - 1)[:k]
            neighbors = candidates[chosen]
            d = distance[chosen]
            outcomes = opens[neighbors + 3] / opens[neighbors + 1] - 1.0
            weights = np.exp(-d / max(float(d.mean()), 0.03))
            weights /= weights.sum()
            edge[i] = float(np.dot(weights, outcomes))
            wins[i] = float(np.dot(weights, outcomes > 0))
            dispersion[i] = float(np.sqrt(np.dot(weights, (outcomes - edge[i]) ** 2)))
            similarity[i] = float(np.exp(-d.mean()))
            count[i] = k
        out["edge"] = edge
        out["win_rate"] = wins
        out["dispersion"] = dispersion
        out["similarity"] = similarity
        out["analog_count"] = count
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.20, dtype=float)
        edge = indicators["edge"].fillna(0.0).to_numpy(dtype=float)
        dispersion = indicators["dispersion"].fillna(0.0).to_numpy(dtype=float)
        wins = indicators["win_rate"].fillna(0.0).to_numpy(dtype=float)
        similarity = indicators["similarity"].fillna(0.0).to_numpy(dtype=float)
        qualified = (
            (indicators["analog_count"] >= max(3, params.neighbors))
            & (indicators["drift"] < -0.10)
            & (indicators["rebound"] > 0.05)
            & (indicators["edge"] > params.min_edge)
            & (indicators["win_rate"] >= 0.55)
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        strength = (
            np.clip(edge / np.maximum(dispersion, 0.001), 0.0, 1.0)
            * np.clip((wins - 0.5) / 0.5, 0.0, 1.0)
            * np.clip(similarity, 0.0, 1.0)
        )
        desired_size = 0.20 + 0.75 * strength
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        active = False
        armed = False
        entry = 0.0
        stop = 0.0
        high_water = 0.0
        entry_bar = -1
        held_size = 0.20
        for i in range(n):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and signal[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    armed = False
                    entry = float(opens[i])
                    high_water = entry
                    stop = entry - params.trail_atr * float(atr[i - 1])
                    entry_bar = i
                    held_size = float(sizes[i - 1])
            if active:
                sizes[i] = held_size
                valid_bar = np.isfinite([highs[i], lows[i], closes[i], atr[i]]).all()
                exit_now = not valid_bar or i - entry_bar + 1 >= 2
                if valid_bar:
                    # Test the carried stop before updating it with today's high.
                    exit_now = exit_now or lows[i] <= stop
                    high_water = max(high_water, float(highs[i]))
                    if highs[i] >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, high_water - params.trail_atr * float(atr[i]))
                    exit_now = exit_now or closes[i] <= stop
                if exit_now:
                    active = False
                else:
                    signal[i] = 1
                # Exit decisions execute next open; no same-close reentry.
                continue
            if qualified[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signal[i] = 1
                sizes[i] = float(desired_size[i])
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
