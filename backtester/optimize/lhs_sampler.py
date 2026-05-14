from __future__ import annotations

import math
import random
from typing import Any


def sample_param_space(
    *,
    space: dict[str, list[Any]],
    random_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Discrete Latin-hypercube sampler over index positions in each candidate list.

    For each parameter with k candidates and a budget of random_n samples:
      - Partition [0, k) into random_n strata of equal width.
      - Draw one float per stratum (uniform within the stratum).
      - Floor to integer index -> maps each stratum to one of the k candidates.
    Then randomly permute each per-parameter index list to decorrelate across
    parameters.

    Each candidate value is selected ~ random_n / k times.

    Reproducible via `seed`. Raises ValueError if random_n exceeds the
    Cartesian product size.
    """
    cartesian_size = 1
    for v in space.values():
        cartesian_size *= len(v)
    # Only enforce the constraint if we have multiple parameters.
    # For a single parameter, oversampling is fine (each value appears multiple times).
    if len(space) > 1 and random_n > cartesian_size:
        raise ValueError(
            f"random_n={random_n} exceeds Cartesian product size {cartesian_size}; "
            f"use sampling='grid' for full enumeration."
        )

    rng = random.Random(seed)
    per_param_indices: dict[str, list[int]] = {}
    for name, values in space.items():
        k = len(values)
        stratum_width = k / random_n
        indices: list[int] = []
        for j in range(random_n):
            u = rng.random()
            x = (j + u) * stratum_width
            idx = min(int(math.floor(x)), k - 1)
            indices.append(idx)
        rng.shuffle(indices)
        per_param_indices[name] = indices

    samples: list[dict[str, Any]] = []
    for j in range(random_n):
        sample = {name: space[name][per_param_indices[name][j]] for name in space}
        samples.append(sample)
    return samples
