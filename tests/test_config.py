from pathlib import Path

from acpa_gemma.config import load_config, get_api_key


def test_loads_api_key_from_secret_config(tmp_path: Path):
    app = tmp_path / "app.toml"
    secrets = tmp_path / "secrets.toml"
    app.write_text(
        """
[gemma]
model = "gemma-4-31b-it"
""".strip(),
        encoding="utf-8",
    )
    secrets.write_text(
        """
[gemma]
api_key = "test-key"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_paths=[app], secret_paths=[secrets])

    assert config.gemma.model == "gemma-4-31b-it"
    assert get_api_key(config) == "test-key"
    assert str(app) in config.loaded_files
    assert str(secrets) in config.loaded_files
