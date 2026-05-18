# Factory `--max-cycles` CLI flag — design

**Date:** 2026-05-19
**Status:** Approved

## Problem

When running the factory loop (`python -m factory.loop`), there is no way to
say "generate N strategies and stop" from the command line. The only controls
are `[loop] max_cycles` in `settings.toml` (which requires editing a tracked
file) and `Ctrl-C`. The user wants to pick the cycle count at launch time.

## Solution

`run_loop()` already accepts a `max_cycles_override` parameter and resolves
precedence (override wins over `settings.loop.max_cycles`). The only gap is
that `factory.loop`'s `main()` never sets it. Expose it as a CLI flag.

### CLI surface

Add one argument to the `factory.loop` argument parser:

```
--max-cycles N    Stop after N cycles (strategy attempts). 0 = unlimited.
                  Overrides [loop] max_cycles in settings.toml.
```

- Type `int`, default `None`. Absent flag → `None` → `run_loop` falls back to
  `settings.loop.max_cycles`, so existing behaviour is unchanged.
- `0` means unlimited, consistent with the existing `[loop] max_cycles`
  setting.
- A negative value is rejected with a clear argparse error before the loop
  starts.

### Count semantics

One cycle is one strategy attempt. `--max-cycles N` runs exactly N cycles and
then stops, regardless of how many of those cycles produced a `complete`
result versus a `failed` one. This matches how `[loop] max_cycles` and
`run_loop`'s loop-termination check already behave.

### Wiring

`main()` passes `max_cycles_override=args.max_cycles` into the existing
`run_loop(...)` call. `run_loop` itself is unchanged — it already handles
`None` (use settings) versus an integer (override).

### Precedence

CLI flag > `settings.toml`. This is already how `run_loop` resolves the two
sources; the flag simply makes the override reachable from the command line.

## Out of scope

- "Run until N *completed* strategies" (skipping failures) — the user
  explicitly chose N attempts.
- Any change to `run_loop`'s signature or termination logic.
- Renaming the existing `[loop] max_cycles` setting or `max_cycles_override`
  parameter.

## Testing

- `main(["--max-cycles", "3"])` stops the loop after 3 cycles, with the
  per-cycle work stubbed.
- A negative value (`--max-cycles -1`) exits with a non-zero status and an
  error message, without starting the loop.

## Affected files

- `factory/loop.py` — add the argparse argument, pass it to `run_loop`.
- `factory/README.md` — one line in the single-machine quickstart next to the
  existing `--seed` / `--settings` note.
- `factory/tests/` — the two tests above.
