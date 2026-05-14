from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class GeneratedParams:
    size: float = 1.0

# Note: no GeneratedStrategy class.
