from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class FilesystemError(RuntimeError):
    pass


class RegistryAlreadyHasStrategy(FilesystemError):
    pass


def pick_unused_strategy_id(base: str, *, strategies_dir: Path) -> str:
    """Return `base` if strategies/<base>.py is free, otherwise base_2, base_3, ..."""
    if not (strategies_dir / f"{base}.py").exists():
        return base
    i = 2
    while (strategies_dir / f"{base}_{i}.py").exists():
        i += 1
    return f"{base}_{i}"


def write_strategy_artifacts(
    *,
    strategy_id: str,
    strategy_src: str,
    config_src: str,
    strategies_dir: Path,
    configs_dir: Path,
) -> tuple[Path, Path]:
    """Write the strategy .py and config .yaml.

    Refuses to overwrite either file (collision should have been avoided by
    pick_unused_strategy_id upstream).
    """
    strat_path = strategies_dir / f"{strategy_id}.py"
    cfg_path = configs_dir / f"{strategy_id}.yaml"
    if strat_path.exists():
        raise FilesystemError(f"strategy file already exists: {strat_path}")
    if cfg_path.exists():
        raise FilesystemError(f"config file already exists: {cfg_path}")
    strategies_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    strat_path.write_text(strategy_src, encoding="utf-8")
    cfg_path.write_text(config_src, encoding="utf-8")
    log.info("wrote strategy=%s config=%s", strat_path, cfg_path)
    return strat_path, cfg_path


def append_registry_entry(*, strategy_id: str, registry_file: Path) -> None:
    """Append two lines to registry.py:
        from strategies.<strategy_id> import GeneratedStrategy as _<strategy_id>
        register_strategy(_<strategy_id>)
    Idempotency: raises RegistryAlreadyHasStrategy if the alias appears already.
    """
    if not registry_file.exists():
        raise FilesystemError(f"registry file not found: {registry_file}")
    text = registry_file.read_text(encoding="utf-8")
    alias = f"_{strategy_id}"
    needle_register = f"register_strategy({alias})"
    if needle_register in text:
        raise RegistryAlreadyHasStrategy(
            f"registry already has strategy {strategy_id!r}"
        )
    if not text.endswith("\n"):
        text += "\n"
    lines = (
        f"from strategies.{strategy_id} import GeneratedStrategy as {alias}  # noqa: E402\n"
        f"register_strategy({alias})\n"
    )
    registry_file.write_text(text + lines, encoding="utf-8")
    log.info("appended registry entry for %s", strategy_id)
