"""Config-file loading for the Gemma Trust & Safety pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_APP_PATHS = [
    Path("configs/app.toml"),
    Path("/kaggle/working/configs/app.toml"),
    Path.home() / ".config/acpa_gemma/config.toml",
]

DEFAULT_SECRET_PATHS = [
    Path("configs/secrets.toml"),
    Path("/kaggle/working/configs/secrets.toml"),
    Path.home() / ".config/acpa_gemma/secrets.toml",
]


@dataclass
class GemmaConfig:
    model: str = "gemma-4-26b-a4b-it"
    api_key: str = ""
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 2048


@dataclass
class DataConfig:
    input_dir: str = "/kaggle/input/agentic-eval"
    sample_size: int = 0


@dataclass
class PruningConfig:
    alpha: float = 1.5
    beta: float = 1.0
    gamma: float = 0.5
    delta: float = 10.0
    prune_ratio: float = 0.45
    cache_threshold: int = 2
    priority_boost: float = 1.5


@dataclass
class OutputConfig:
    path: str = "/kaggle/working/results.jsonl"


@dataclass
class AppConfig:
    gemma: GemmaConfig = field(default_factory=GemmaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    loaded_files: List[str] = field(default_factory=list)


def load_config(
    config_paths: Optional[Iterable[str | Path]] = None,
    secret_paths: Optional[Iterable[str | Path]] = None,
) -> AppConfig:
    """Load application config and secrets from TOML files.

    Later files override earlier values. Secret files are loaded last so an API
    key in `configs/secrets.toml` takes precedence over non-secret defaults.
    """

    app_data: Dict[str, Any] = {}
    loaded_files: List[str] = []
    for path in _existing_paths(config_paths or DEFAULT_APP_PATHS):
        deep_update(app_data, read_toml(path))
        loaded_files.append(str(path))

    for path in _existing_paths(secret_paths or DEFAULT_SECRET_PATHS):
        deep_update(app_data, read_toml(path))
        loaded_files.append(str(path))

    config = AppConfig(
        gemma=_build_dataclass(GemmaConfig, app_data.get("gemma", {})),
        data=_build_dataclass(DataConfig, app_data.get("data", {})),
        pruning=_build_dataclass(PruningConfig, app_data.get("pruning", {})),
        output=_build_dataclass(OutputConfig, app_data.get("output", {})),
        loaded_files=loaded_files,
    )
    return config


def get_api_key(config: AppConfig) -> str:
    """Return the Gemma API key from loaded config files."""

    return config.gemma.api_key.strip()


def read_toml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def deep_update(target: Dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def _existing_paths(paths: Iterable[str | Path]) -> List[Path]:
    return [Path(path) for path in paths if Path(path).exists()]


def _build_dataclass(cls: type, values: Mapping[str, Any]):
    allowed = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in values.items() if key in allowed})
