"""Tests for ``pyiwfm.io.cache_paths`` (cache directory resolution + migration)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.cache_paths import (
    ENV_VAR,
    SQLITE_CACHE_NAME,
    migrate_legacy_caches,
    resolve_cache_dir,
)


class TestResolveCacheDir:
    def test_override_wins_over_env_and_results_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "from_env"))
        results = tmp_path / "results"
        results.mkdir()
        override = tmp_path / "from_override"

        resolved = resolve_cache_dir(results_dir=results, override=override)

        assert resolved == override
        assert resolved.is_dir()

    def test_env_var_wins_over_results_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_target = tmp_path / "from_env"
        monkeypatch.setenv(ENV_VAR, str(env_target))
        results = tmp_path / "results"
        results.mkdir()

        resolved = resolve_cache_dir(results_dir=results)

        assert resolved == env_target
        assert resolved.is_dir()

    def test_results_dir_default_subfolder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)
        results = tmp_path / "results"
        results.mkdir()

        resolved = resolve_cache_dir(results_dir=results)

        assert resolved == results / "pyiwfm_cache"
        assert resolved.is_dir()

    def test_falls_back_to_source_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)
        source = tmp_path / "source"
        source.mkdir()

        resolved = resolve_cache_dir(results_dir=None, source_dir=source)

        assert resolved == source / "pyiwfm_cache"
        assert resolved.is_dir()

    def test_returns_none_when_nothing_to_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR, raising=False)

        assert resolve_cache_dir(results_dir=None, source_dir=None) is None


class TestMigrateLegacyCaches:
    def test_migrates_sqlite_and_area_caches(self, tmp_path: Path) -> None:
        legacy = tmp_path / "results"
        legacy.mkdir()
        cache_dir = legacy / "pyiwfm_cache"
        cache_dir.mkdir()

        (legacy / SQLITE_CACHE_NAME).write_bytes(b"old-sqlite")
        (legacy / "nonponded_area_cache.hdf").write_bytes(b"npa")
        (legacy / "ponded_area_cache.hdf").write_bytes(b"pa")

        moved = migrate_legacy_caches(cache_dir, legacy_dirs=[legacy])

        assert moved == 3
        assert (cache_dir / SQLITE_CACHE_NAME).read_bytes() == b"old-sqlite"
        assert (cache_dir / "nonponded_area_cache.hdf").read_bytes() == b"npa"
        assert not (legacy / SQLITE_CACHE_NAME).exists()
        assert not (legacy / "nonponded_area_cache.hdf").exists()

    def test_skips_when_target_already_exists(self, tmp_path: Path) -> None:
        legacy = tmp_path / "results"
        legacy.mkdir()
        cache_dir = legacy / "pyiwfm_cache"
        cache_dir.mkdir()

        (legacy / SQLITE_CACHE_NAME).write_bytes(b"old")
        (cache_dir / SQLITE_CACHE_NAME).write_bytes(b"new")

        moved = migrate_legacy_caches(cache_dir, legacy_dirs=[legacy])

        assert moved == 0
        # Legacy file untouched (user can inspect/delete it manually).
        assert (legacy / SQLITE_CACHE_NAME).read_bytes() == b"old"
        assert (cache_dir / SQLITE_CACHE_NAME).read_bytes() == b"new"

    def test_migrates_head_and_hydrograph_caches(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        cache_dir = results / "pyiwfm_cache"
        cache_dir.mkdir()

        head_src = results / "GWHead.out"
        head_src.write_text("not-real-head-data")
        legacy_head_cache = head_src.with_suffix(".head_cache.hdf")
        legacy_head_cache.write_bytes(b"head-cache")

        hydro_src = results / "GWHydro.out"
        hydro_src.write_text("not-real-hydro-data")
        legacy_hydro_cache = hydro_src.parent / (hydro_src.name + ".hydrograph_cache.hdf")
        legacy_hydro_cache.write_bytes(b"hydro-cache")

        moved = migrate_legacy_caches(
            cache_dir,
            legacy_dirs=[],
            head_source_paths=[head_src],
            hydrograph_source_paths=[hydro_src],
        )

        assert moved == 2
        assert (cache_dir / "GWHead.head_cache.hdf").read_bytes() == b"head-cache"
        assert (cache_dir / "GWHydro.out.hydrograph_cache.hdf").read_bytes() == b"hydro-cache"
        assert not legacy_head_cache.exists()
        assert not legacy_hydro_cache.exists()

    def test_no_op_when_no_legacy_files(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "pyiwfm_cache"
        cache_dir.mkdir()

        moved = migrate_legacy_caches(cache_dir, legacy_dirs=[tmp_path / "missing"])

        assert moved == 0
