from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.wfo.multi_runner import WindowResult


@dataclass
class StitchedWFOResult:
    oos_equity_curve: pd.Series
    oos_summary: dict[str, float]
    window_summaries: list[dict]
    parameter_stability: dict[str, list]


@dataclass(slots=True)
class MultiSymbolWFOStitcher:
    """Combine per-window OOS equity curves into a single continuous portfolio equity series.

    Each window's OOS equity curve starts at some initial cash value. The stitcher
    scales each subsequent window so that the first OOS bar of window N+1 picks up
    where the last bar of window N's OOS left off.
    """

    def stitch(self, window_results: list[WindowResult]) -> StitchedWFOResult:
        if not window_results:
            raise ValueError("no window results to stitch")

        stitched_pieces: list[pd.Series] = []
        running_equity = None
        for wr in window_results:
            eq = wr.oos_equity_curve.copy()
            if running_equity is None:
                stitched_pieces.append(eq)
                running_equity = float(eq.iloc[-1])
            else:
                # Scale this window's curve so its first bar matches running_equity.
                scale = running_equity / float(eq.iloc[0])
                scaled = eq * scale
                stitched_pieces.append(scaled)
                running_equity = float(scaled.iloc[-1])

        oos_curve = pd.concat(stitched_pieces).sort_index()
        # Drop duplicates in case adjacent windows overlap on boundary.
        oos_curve = oos_curve[~oos_curve.index.duplicated(keep="first")]

        # Compute oos summary.
        if len(oos_curve) > 1 and oos_curve.iloc[0] > 0:
            returns = oos_curve.pct_change().dropna()
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
            peak = oos_curve.cummax()
            drawdown = (oos_curve - peak) / peak
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
            total_return = float(oos_curve.iloc[-1] / oos_curve.iloc[0] - 1.0)
        else:
            sharpe = max_dd = total_return = 0.0

        oos_summary = {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "calmar": (total_return / abs(max_dd)) if max_dd < 0 else 0.0,
            "n_windows": len(window_results),
        }

        window_summaries = [
            {
                "window_idx": wr.window_idx,
                "train_start": str(wr.train_start.date()),
                "train_end": str(wr.train_end.date()),
                "test_start": str(wr.test_start.date()),
                "test_end": str(wr.test_end.date()),
                "best_params": (
                    wr.best_params.__dict__ if hasattr(wr.best_params, "__dict__")
                    else dict(wr.best_params)
                ),
                "is_summary": wr.is_summary,
                "oos_summary": wr.oos_summary,
            }
            for wr in window_results
        ]

        # Parameter stability: dict[param_name, list of best values per window]
        parameter_stability: dict[str, list] = {}
        for wr in window_results:
            params_dict = (
                wr.best_params.__dict__ if hasattr(wr.best_params, "__dict__")
                else dict(wr.best_params)
            )
            for k, v in params_dict.items():
                parameter_stability.setdefault(k, []).append(v)

        return StitchedWFOResult(
            oos_equity_curve=oos_curve,
            oos_summary=oos_summary,
            window_summaries=window_summaries,
            parameter_stability=parameter_stability,
        )
