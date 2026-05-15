# Strategy Factory v0.2.0

Local, unattended strategy-idea factory wrapping the Modular-Backtester.

## Quickstart

1. `pip install -e .[dev]` from the backtester repo root (installs the backtester package and pytest).
2. `pip install Flask` (the factory's only extra runtime dep).
3. Edit `factory/config/settings.toml` — set `backtester_root` to an absolute path if needed, and fill `telegram_bot_token` / `telegram_chat_id` if you want alerts.
4. Run the loop: `python -m factory.loop`
5. Run the dashboard (separate terminal): `python -m factory.dashboard.server`
6. Open `http://127.0.0.1:8787`

## Tests

`python -m pytest factory/tests -q` from the backtester root.

Slow tests (Tier 2 functional validation, integration smoke) are marked `@pytest.mark.slow`. Run with `-m slow` to include them, `-m "not slow"` to skip.

## Spec

See `docs/superpowers/plans/2026-05-15-strategy-factory-v020.md` for the implementation plan and the linked spec.
