from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.4
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791339736"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 694

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.0 < params.frac_order < 1.0:
            raise ValueError("frac_order must lie between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.ones(64)
        for lag in range(1, 64):
            weights[lag] = weights[lag - 1] * (lag - 1 - params.frac_order) / lag
        # ponytail: 64-lag fractional memory; extend only if truncation matters.
        out["frac_close"] = sum(weights[lag] * close.shift(lag) for lag in range(64))
        out["frac_mean"] = out["frac_close"].rolling(126).mean()
        out["frac_std"] = out["frac_close"].rolling(126).std()
        out["frac_z"] = (out["frac_close"] - out["frac_mean"]) / out["frac_std"].replace(0.0, np.nan)
        out["frac_change"] = out["frac_z"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14).mean()
        out["phase"] = 2.0 * np.pi * (data.index.day - 1) / data.index.days_in_month
        out["cos_phase"] = np.cos(out["phase"])
        out["sin_phase"] = np.sin(out["phase"])
        # Fit only complete oscillator observations, using past bars.
        c = out["cos_phase"].where(out["frac_z"].notna())
        s = out["sin_phase"].where(out["frac_z"].notna())
        z = out["frac_z"]
        out["var_c"] = c.rolling(504).var()
        out["var_s"] = s.rolling(504).var()
        out["cov_cs"] = c.rolling(504).cov(s)
        out["cov_zc"] = z.rolling(504).cov(c)
        out["cov_zs"] = z.rolling(504).cov(s)
        out["det"] = out["var_c"] * out["var_s"] - out["cov_cs"] ** 2
        denominator = out["det"].where(out["det"] > 1e-12)
        out["coef_c"] = ((out["cov_zc"] * out["var_s"] - out["cov_zs"] * out["cov_cs"]) / denominator).shift(1)
        out["coef_s"] = ((out["cov_zs"] * out["var_c"] - out["cov_zc"] * out["cov_cs"]) / denominator).shift(1)
        out["season_now"] = out["coef_c"] * out["cos_phase"] + out["coef_s"] * out["sin_phase"]
        # Eight sessions are approximately ten calendar days.
        out["future_phase"] = out["phase"] + 2.0 * np.pi * 10.0 / 30.4375
        out["season_future"] = out["coef_c"] * np.cos(out["future_phase"]) + out["coef_s"] * np.sin(out["future_phase"])
        out["season_gain"] = out["season_future"] - out["season_now"]
        out["entry"] = (
            (out["frac_z"] < 0.0)
            & (out["frac_change"] > 0.0)
            & (out["season_future"] > 0.0)
            & (out["season_gain"] > 0.15)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    entry_bar = i
                    # Entry open is observed only on the actual entry bar.
                    # ATR was frozen at the preceding decision close.
                    stop = opens[i] - params.stop_atr * entry_atr
            if active:
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar >= 7:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(closes[i]):
                signal[i] = 1
                pending = True
                entry_atr = atr[i]
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    ind = pd.DataFrame({"entry": False, "atr": 1.0}, index=index)
    ind.iloc[0, 0] = True
    data.iloc[1, 1] = 104.0
    data.iloc[2, 1] = 99.0
    data.iloc[3, 1] = 97.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
