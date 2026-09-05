"""Cấu hình: fail-fast, báo rõ biến nào thiếu, không rò secret ra log."""

from __future__ import annotations

from typing import Any

import pytest

from rag_core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_settings.cache_clear()


def _settings(**overrides: Any) -> Settings:
    # `_env_file=None` để test không phụ thuộc `.env` của máy đang chạy.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestDefaults:
    def test_loads_without_any_env(self) -> None:
        settings = _settings()
        assert settings.qdrant_url == "http://127.0.0.1:6333"
        assert settings.deepseek_api_key is None

    def test_reads_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_COLLECTION", "custom_collection")
        assert _settings().qdrant_collection == "custom_collection"

    def test_rejects_unknown_device(self) -> None:
        with pytest.raises(ValueError, match="EMBEDDING_DEVICE"):
            _settings(embedding_device="tpu")


class TestSecretHandling:
    def test_secret_not_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Log nguyên object settings là chuyện xảy ra thường xuyên lúc debug —
        khi đó key không được lộ ra."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-that-la-bi-mat")
        settings = _settings()
        assert "sk-that-la-bi-mat" not in repr(settings)
        assert "sk-that-la-bi-mat" not in str(settings)
        assert settings.deepseek_api_key is not None
        assert settings.deepseek_api_key.get_secret_value() == "sk-that-la-bi-mat"

    def test_password_not_in_repr_but_in_dsn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "mat-khau")
        settings = _settings()
        assert "mat-khau" not in repr(settings)
        assert "mat-khau" in settings.postgres_dsn

    def test_a_password_with_at_sign_does_not_change_the_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`NEW-08`/`AU-04`: mật khẩu `s3cret@evil.com/db` trong f-string trần
        làm SQLAlchemy parse `evil.com` thành HOST — kết nối (kể cả DSN
        migration mang quyền superuser) đi sang máy khác mà không lỗi nào
        cảnh báo. `quote_plus` biến mọi ký tự cấu trúc URL thành dữ liệu."""
        from sqlalchemy.engine.url import make_url

        monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret@evil.com/db")
        monkeypatch.setenv("POSTGRES_HOST", "real-host")
        settings = _settings()

        url = make_url(settings.postgres_dsn)
        assert url.host == "real-host"
        assert url.password == "s3cret@evil.com/db", "mật khẩu phải sống sót một vòng parse"

    def test_a_user_with_url_characters_survives_the_roundtrip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sqlalchemy.engine.url import make_url

        monkeypatch.setenv("POSTGRES_APP_USER", "svc:rag@prod")
        monkeypatch.setenv("POSTGRES_APP_PASSWORD", "p%40ss:w/ord")
        settings = _settings()

        url = make_url(settings.postgres_app_dsn)
        assert url.username == "svc:rag@prod"
        assert url.password == "p%40ss:w/ord"


class TestRequire:
    def test_passes_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        _settings().require("deepseek_api_key")

    def test_reports_all_missing_at_once(self) -> None:
        # Báo từng biến một khiến người dùng phải chạy lại 3 lần mới biết đủ.
        with pytest.raises(RuntimeError) as excinfo:
            _settings().require("deepseek_api_key", "openrouter_api_key")
        message = str(excinfo.value)
        assert "DEEPSEEK_API_KEY" in message
        assert "OPENROUTER_API_KEY" in message
        assert ".env.example" in message


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
