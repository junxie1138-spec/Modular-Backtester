from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from backtester.analytics.metrics import compute_summary_metrics


class WalkForwardStitcher:
    """Concatenate OOS equity curves and recompute summary metrics across the stitched series."""

    def combine(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not window_results:
            raise ValueError("stitcher received no windows")

        oos_pieces = []
        oos_trades = []
        oos_positions = []
        prev_end = None

        for w in window_results:
            eq = w["test_result"].equity_curve.copy()
            # Re-base each window's equity onto the running OOS equity series
            if prev_end is None:
                scale = 1.0
            else:
                scale = prev_end / eq["equity"].iloc[0]
            eq["equity"] = eq["equity"] * scale
            if "cash" in eq.columns:
                eq["cash"] = eq["cash"] * scale
            if "position_value" in eq.columns:
                eq["position_value"] = eq["position_value"] * scale
            oos_pieces.append(eq)
            prev_end = eq["equity"].iloc[-1]

            oos_trades.append(w["test_result"].trades)
            oos_positions.append(w["test_result"].positions)

        oos_eq = pd.concat(oos_pieces).sort_index()
        # de-duplicate index if windows abut
        oos_eq = oos_eq[~oos_eq.index.duplicated(keep="last")]
        oos_trades_df = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
        oos_positions_df = pd.concat(oos_positions) if oos_positions else pd.DataFrame()

        oos_summary = compute_summary_metrics(oos_eq, oos_trades_df, oos_positions_df)

        # IS averages
        is_summaries = [w["train_summary"] for w in window_results]
        is_keys = set().union(*[set(s.keys()) for s in is_summaries])
        is_avg = {k: float(np.mean([float(s.get(k, 0.0)) for s in is_summaries if isinstance(s.get(k, 0), (int, float))]))
                  for k in is_keys}

        # parameter stability
        stability: Dict[str, Dict[str, Any]] = {}
        all_keys = set().union(*[set(w["best_params"].keys()) for w in window_results])
        for k in all_keys:
            values = [w["best_params"].get(k) for w in window_results]
            counter = Counter(values)
            stability[k] = {
                "unique": len(set(values)),
                "mode": counter.most_common(1)[0][0],
                "values_by_window": values,
            }

        return {
            "oos_equity_curve": oos_eq,
            "oos_trades": oos_trades_df,
            "oos_positions": oos_positions_df,
            "oos_summary": oos_summary,
            "is_summary_avg": is_avg,
            "parameter_stability": stability,
            "window_results": [
                {k: v for k, v in w.items() if k != "test_result"}
                for w in window_results
            ],
        }
