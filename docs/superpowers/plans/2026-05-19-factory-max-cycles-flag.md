# Factory `--max-cycles` CLI Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator pick how many cycles the factory runs by passing `--max-cycles N` to `python -m factory.loop`.

**Architecture:** `run_loop()` already accepts a `max_cycles_override` parameter and resolves precedence (override beats `settings.loop.max_cycles`). The only gap is that `factory.loop`'s `main()` never sets it. This plan adds one argparse argument, a validator that rejects negative values, and wires the parsed value into the existing `run_loop(...)` call. No change to `run_loop` itself.

**Tech Stack:** Python 3.11+, `argparse`, `pytest`, `unittest.mock`.

---

### Task 1: Add the `--max-cycles` flag and wire it into `run_loop`

**Files:**
- Modify: `factory/loop.py` (the `main` function, lines 152-171)
- Test: `factory/tests/test_loop.py`

The `tmp_settings_file` fixture (in `factory/tests/conftest.py`) writes a complete settings.toml with `[loop] max_cycles = 1`. Tests mock `factory.loop.run_loop` so the loop never actually runs, and mock `factory.loop.configure_logging` so no log file is created.

- [ ] **Step 1: Write the failing tests**

Add these two tests to the end of `factory/tests/test_loop.py`:

```python
def test_main_passes_max_cycles_override(tmp_settings_file: Path) -> None:
    with mock.patch("factory.loop.run_loop") as rl, \
         mock.patch("factory.loop.configure_logging"):
        from factory.loop import main
        rc = main(["--settings", str(tmp_settings_file), "--max-cycles", "3"])
    assert rc == 0
    assert rl.call_args.kwargs["max_cycles_override"] == 3


def test_main_max_cycles_defaults_to_none(tmp_settings_file: Path) -> None:
    with mock.patch("factory.loop.run_loop") as rl, \
         mock.patch("factory.loop.configure_logging"):
        from factory.loop import main
        main(["--settings", str(tmp_settings_file)])
    assert rl.call_args.kwargs["max_cycles_override"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest factory/tests/test_loop.py::test_main_passes_max_cycles_override factory/tests/test_loop.py::test_main_max_cycles_defaults_to_none -v`

Expected: both FAIL — `test_main_passes_max_cycles_override` with a `KeyError: 'max_cycles_override'` (the kwarg is not passed), and `test_main_max_cycles_defaults_to_none` likewise. (If `main` does not parse `--max-cycles` at all, the first test fails earlier with a `SystemExit` from argparse rejecting the unknown flag — also a valid failure.)

- [ ] **Step 3: Add the argument and wire it through**

In `factory/loop.py`, inside `main()`, add the new argument to the parser. The current parser block is:

```python
    parser = argparse.ArgumentParser("factory.loop")
    parser.add_argument(
        "--settings",
        default="factory/config/settings.toml",
        type=Path,
        help="Path to settings.toml",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for slot pulls")
    args = parser.parse_args(argv)
```

Replace it with:

```python
    parser = argparse.ArgumentParser("factory.loop")
    parser.add_argument(
        "--settings",
        default="factory/config/settings.toml",
        type=Path,
        help="Path to settings.toml",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for slot pulls")
    parser.add_argument(
        "--max-cycles", type=int, default=None,
        help="Stop after N cycles (strategy attempts). 0 = unlimited. "
             "Overrides [loop] max_cycles in settings.toml.",
    )
    args = parser.parse_args(argv)
```

Then change the `run_loop(...)` call. The current line is:

```python
    run_loop(s, rng=rng, shutdown_flag=flag)
```

Replace it with:

```python
    run_loop(s, rng=rng, shutdown_flag=flag,
             max_cycles_override=args.max_cycles)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest factory/tests/test_loop.py::test_main_passes_max_cycles_override factory/tests/test_loop.py::test_main_max_cycles_defaults_to_none -v`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add factory/loop.py factory/tests/test_loop.py
git commit -m "feat: add --max-cycles flag to factory loop CLI"
```

---

### Task 2: Reject a negative `--max-cycles` value

**Files:**
- Modify: `factory/loop.py` (the `main` function — replace `type=int` on the new argument)
- Test: `factory/tests/test_loop.py`

`argparse` with `type=int` accepts `-1`. A negative cycle count is meaningless, so the flag needs a validator that rejects it before the loop starts. A custom `type` callable that raises `argparse.ArgumentTypeError` makes argparse exit with status 2 and a clear message.

- [ ] **Step 1: Write the failing test**

Add this test to the end of `factory/tests/test_loop.py`:

```python
def test_main_rejects_negative_max_cycles(tmp_settings_file: Path) -> None:
    from factory.loop import main
    with pytest.raises(SystemExit) as exc:
        main(["--settings", str(tmp_settings_file), "--max-cycles", "-1"])
    assert exc.value.code != 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest factory/tests/test_loop.py::test_main_rejects_negative_max_cycles -v`

Expected: FAIL — `main` does not raise `SystemExit` because `type=int` accepts `-1`; instead it proceeds to call the (unmocked) `run_loop`, so the `pytest.raises(SystemExit)` block is never satisfied.

- [ ] **Step 3: Add the validator and use it**

In `factory/loop.py`, add this module-level helper just above `def main(`:

```python
def _nonneg_int(value: str) -> int:
    """argparse `type` for --max-cycles: a non-negative integer."""
    import argparse
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0 (0 = unlimited)")
    return n
```

Then, in `main()`, change the `--max-cycles` argument's `type` from `int` to `_nonneg_int`. The argument becomes:

```python
    parser.add_argument(
        "--max-cycles", type=_nonneg_int, default=None,
        help="Stop after N cycles (strategy attempts). 0 = unlimited. "
             "Overrides [loop] max_cycles in settings.toml.",
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest factory/tests/test_loop.py::test_main_rejects_negative_max_cycles -v`

Expected: PASS.

- [ ] **Step 5: Run the full loop test file to confirm nothing regressed**

Run: `python -m pytest factory/tests/test_loop.py -v`

Expected: all tests PASS (the three new ones plus the six pre-existing).

- [ ] **Step 6: Commit**

```bash
git add factory/loop.py factory/tests/test_loop.py
git commit -m "feat: reject negative --max-cycles values"
```

---

### Task 3: Document the flag in the factory README

**Files:**
- Modify: `factory/README.md` (the single-machine quickstart, line 55)

- [ ] **Step 1: Update the quickstart line**

In `factory/README.md`, find this line in the "Single-machine quickstart" section:

```markdown
The loop runs until `Ctrl-C` (graceful — it finishes the current cycle), or until `[loop] max_cycles` is reached. `python -m factory.loop --seed 42` makes idea-slot draws reproducible; `--settings <path>` points at an alternate settings file.
```

Replace it with:

```markdown
The loop runs until `Ctrl-C` (graceful — it finishes the current cycle), or until `[loop] max_cycles` is reached. `python -m factory.loop --max-cycles 10` stops after 10 cycles (strategy attempts), overriding `[loop] max_cycles` without editing settings (`0` = unlimited). `python -m factory.loop --seed 42` makes idea-slot draws reproducible; `--settings <path>` points at an alternate settings file.
```

- [ ] **Step 2: Commit**

```bash
git add factory/README.md
git commit -m "docs: document factory --max-cycles flag"
```

---

## Self-Review

**Spec coverage:**
- CLI surface (`--max-cycles N`, type int, default None) — Task 1.
- `0` = unlimited — inherited from `run_loop`'s existing handling; documented in the help text (Task 1) and README (Task 3).
- Negative value rejected with a clear error before the loop starts — Task 2.
- Count semantics (N attempts, not N successes) — no code; the README wording (Task 3) and help text (Task 1) state "strategy attempts". `run_loop`'s termination check is unchanged, so semantics are already correct.
- Wiring `max_cycles_override=args.max_cycles` into the existing `run_loop` call — Task 1.
- Precedence CLI > settings — inherited from `run_loop`; covered by `test_main_max_cycles_defaults_to_none` (absent flag → `None` → settings used) and `test_main_passes_max_cycles_override`.
- `run_loop` signature unchanged — confirmed; no task touches it.
- Tests (stop after N cycles, negative rejected) — Tasks 1 and 2.

**Placeholder scan:** No TBD/TODO/vague steps. Every code step shows complete code.

**Type consistency:** The argparse argument is `--max-cycles`, parsed as `args.max_cycles` (argparse converts the hyphen to an underscore). The validator `_nonneg_int` is defined in Task 2 and referenced only after definition. `run_loop`'s `max_cycles_override` keyword matches its signature in `factory/loop.py:86`.
