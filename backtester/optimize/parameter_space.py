from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterator, List


def expand_grid(space: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """Yield every combination from a parameter grid as dicts."""
    if not space:
        yield {}
        return
    keys = list(space.keys())
    values = [space[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))
