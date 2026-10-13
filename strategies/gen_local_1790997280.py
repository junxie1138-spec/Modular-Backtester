from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    pattern_window: int = 42
    fit_window: int = 252
    entry_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[TideParams]):
    strategy_id = "gen_local_1790997280"

    @classmethod
    def params_type(cls):
        return TideParams

    def warmup_bars(self, params: TideParams) -> int:
        return int(params.pattern_window + params.fit_window + 4)

    def indicators(self, data: pd.DataFrame, params: TideParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        triples = pd.concat([out["return"].shift(2), out["return"].shift(1), out["return"]], axis=1)
        valid = triples.notna().all(axis=1)
        order = np.argsort(triples.to_numpy(), axis=1, kind="stable")
        code = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        out["pattern"] = pd.Series(code, index=data.index).where(valid)
        codes = (5, 7, 11, 15, 19, 21)
        for k in codes:
            out[f"p_{k}"] = out["pattern"].eq(k).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
        probabilities = out[[f"p_{k}" for k in codes]]
        out["entropy"] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(
            axis=1, min_count=6
        ) / np.log(6.0)
        out["disequilibrium"] = ((probabilities - 1.0 / 6.0) ** 2).sum(
            axis=1, min_count=6
        ) * 6.0 / 5.0
        out["complexity"] = out["entropy"] * out["disequilibrium"]
        out["weight"] = 0.2 + out["complexity"].shift(1).clip(0.0, 1.0)
        dates = pd.DatetimeIndex(data.index)
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        days = (dates.normalize() - pd.Timestamp("1970-01-05")).days.to_numpy()
        out["cosine"] = np.cos(2.0 * np.pi * days / 14.0)
        out["sine"] = np.sin(2.0 * np.pi * days / 14.0)
        w, c, s, r = out["weight"], out["cosine"], out["sine"], out["return"]
        raw = pd.DataFrame({
            "w": w, "c": w * c, "s": w * s, "r": w * r,
            "cc": w * c * c, "ss": w * s * s, "cs": w * c * s,
            "cr": w * c * r, "sr": w * s * r,
        }, index=data.index)
        moments = raw.rolling(params.fit_window, min_periods=params.fit_window).sum()
        moments = moments.div(moments["w"].replace(0.0, np.nan), axis=0)
        for name in moments:
            out[f"moment_{name}"] = moments[name]
        out["var_cosine"] = moments["cc"] - moments["c"] ** 2
        out["var_sine"] = moments["ss"] - moments["s"] ** 2
        out["cov_cs"] = moments["cs"] - moments["c"] * moments["s"]
        out["cov_cr"] = moments["cr"] - moments["c"] * moments["r"]
        out["cov_sr"] = moments["sr"] - moments["s"] * moments["r"]
        out["determinant"] = out["var_cosine"] * out["var_sine"] - out["cov_cs"] ** 2
        determinant = out["determinant"].where(out["determinant"] > 1e-12)
        out["beta_cosine"] = (out["cov_cr"] * out["var_sine"] - out["cov_sr"] * out["cov_cs"]) / determinant
        out["beta_sine"] = (out["cov_sr"] * out["var_cosine"] - out["cov_cr"] * out["cov_cs"]) / determinant
        out["future_cosine"] = 0.0
        out["future_sine"] = 0.0
        # Known weekday calendar only; no future prices or data rows are read.
        for step in range(1, 5):
            future_days = ((dates + pd.offsets.BDay(step)).normalize() - pd.Timestamp("1970-01-05")).days.to_numpy()
            out["future_cosine"] += np.cos(2.0 * np.pi * future_days / 14.0)
            out["future_sine"] += np.sin(2.0 * np.pi * future_days / 14.0)
        out["forecast"] = out["beta_cosine"] * out["future_cosine"] + out["beta_sine"] * out["future_sine"]
        out["volatility"] = r.rolling(params.pattern_window, min_periods=params.pattern_window).std(ddof=0)
        out["score"] = (
            out["forecast"] / (2.0 * out["volatility"].where(out["volatility"] > 1e-12))
        ) * (out["complexity"] / 0.10).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideParams,
    ) -> SignalFrame:
        score = indicators["score"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        position = 0
        position_size = 0.2
        for i, value in enumerate(score):
            if np.isfinite(value):
                candidate = 1 if value >= params.entry_threshold else (-1 if value <= -params.entry_threshold else 0)
                if position != 0 and candidate == -position:
                    # Explicit flat exit; the opposite direction may enter on a later bar.
                    position = 0
                    position_size = 0.2
                elif position == 0 and candidate != 0:
                    position = candidate
                    strength = min(abs(value) / (3.0 * params.entry_threshold), 1.0)
                    position_size = 0.2 + 0.75 * strength
            signals[i] = position
            sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
