"""Credential sources never need shell exports or committed secrets."""

import pytest


@pytest.fixture
def settings_root(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "ZAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".streamlit").mkdir()
    return tmp_path


@pytest.mark.parametrize("name", ["HF_TOKEN", "ZAI_API_KEY"])
def test_project_file_supplies_keys(settings_root, name):
    from fintech_support.settings import get_secret

    (settings_root / ".streamlit/secrets.toml").write_text(f'{name} = "test-only-value"\n')
    assert get_secret(name, root=settings_root) == "test-only-value"


def test_environment_overrides_file(settings_root, monkeypatch):
    from fintech_support.settings import get_secret

    (settings_root / ".streamlit/secrets.toml").write_text('HF_TOKEN = "file-value"\n')
    monkeypatch.setenv("HF_TOKEN", "environment-value")
    assert get_secret("HF_TOKEN", root=settings_root) == "environment-value"


def test_project_overrides_global(settings_root):
    from fintech_support.settings import get_secret

    global_dir = settings_root / "home/.streamlit"
    global_dir.mkdir(parents=True)
    (global_dir / "secrets.toml").write_text('HF_TOKEN = "global-value"\nZAI_API_KEY = "global-zai"\n')
    (settings_root / ".streamlit/secrets.toml").write_text('HF_TOKEN = "project-value"\n')
    assert get_secret("HF_TOKEN", root=settings_root) == "project-value"
    assert get_secret("ZAI_API_KEY", root=settings_root) == "global-zai"


def test_huggingface_alias_and_missing_key(settings_root):
    from fintech_support.settings import get_secret

    (settings_root / ".streamlit/secrets.toml").write_text('HUGGINGFACE_HUB_TOKEN = "alias-value"\n')
    assert get_secret("HF_TOKEN", root=settings_root, aliases=("HUGGINGFACE_HUB_TOKEN",)) == "alias-value"
    assert get_secret("ZAI_API_KEY", root=settings_root) is None


def test_invalid_file_error_omits_credential_contents(settings_root):
    from fintech_support.settings import get_secret

    (settings_root / ".streamlit/secrets.toml").write_text('HF_TOKEN = "private-test-value')
    with pytest.raises(RuntimeError, match="Invalid secrets file") as error:
        get_secret("HF_TOKEN", root=settings_root)
    assert "private-test-value" not in str(error.value)
