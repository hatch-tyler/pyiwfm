"""Tests for the ``pyiwfm cache clear`` subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pyiwfm.cli.cache import _collect_targets, add_cache_parser, run_cache_clear


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {"cache_dir": None, "model_dir": None, "dry_run": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _seed_cache(cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = [
        cache_dir / "model_cache.db",
        cache_dir / "nonponded_area_cache.hdf",
        cache_dir / "GWHead.head_cache.hdf",
        cache_dir / "GWHydro.out.hydrograph_cache.hdf",
    ]
    for f in files:
        f.write_bytes(b"x")
    return files


class TestCollectTargets:
    def test_explicit_cache_dir(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "pyiwfm_cache"
        seeded = _seed_cache(cache_dir)
        # Drop a non-cache file to confirm we ignore it.
        (cache_dir / "README.txt").write_text("not a cache file")

        targets = _collect_targets(cache_dir_arg=cache_dir, model_dir_arg=None)

        assert sorted(targets) == sorted(seeded)

    def test_model_dir_finds_cache_subfolder_and_legacy(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "Results"
        cache_dir = results_dir / "pyiwfm_cache"
        seeded = _seed_cache(cache_dir)

        # Orphaned legacy in-place files left over from before the refactor.
        legacy_db = results_dir / "model_cache.db"
        legacy_db.write_bytes(b"old")
        legacy_head = results_dir / "OldHead.head_cache.hdf"
        legacy_head.write_bytes(b"old-head")

        targets = _collect_targets(cache_dir_arg=None, model_dir_arg=tmp_path)

        assert legacy_db in targets
        assert legacy_head in targets
        for f in seeded:
            assert f in targets

    def test_env_var_used_when_no_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "from_env"
        seeded = _seed_cache(cache_dir)
        monkeypatch.setenv("PYIWFM_CACHE_DIR", str(cache_dir))

        targets = _collect_targets(cache_dir_arg=None, model_dir_arg=None)

        assert sorted(targets) == sorted(seeded)

    def test_explicit_cache_dir_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = tmp_path / "from_env"
        explicit = tmp_path / "from_arg"
        _seed_cache(env_dir)
        seeded_explicit = _seed_cache(explicit)
        monkeypatch.setenv("PYIWFM_CACHE_DIR", str(env_dir))

        targets = _collect_targets(cache_dir_arg=explicit, model_dir_arg=None)

        assert sorted(targets) == sorted(seeded_explicit)

    def test_returns_empty_when_no_caches(self, tmp_path: Path) -> None:
        targets = _collect_targets(cache_dir_arg=None, model_dir_arg=tmp_path)
        assert targets == []

    def test_warns_on_missing_model_dir(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        targets = _collect_targets(cache_dir_arg=None, model_dir_arg=tmp_path / "does-not-exist")
        assert targets == []
        assert "model directory not found" in capsys.readouterr().out


class TestRunCacheClear:
    def test_dry_run_lists_but_does_not_delete(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "pyiwfm_cache"
        seeded = _seed_cache(cache_dir)

        rc = run_cache_clear(_make_args(cache_dir=cache_dir, dry_run=True))

        assert rc == 0
        assert all(f.exists() for f in seeded)
        out = capsys.readouterr().out
        assert "Would remove" in out
        assert "dry-run" in out

    def test_clear_deletes_files_and_empty_subdir(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "pyiwfm_cache"
        seeded = _seed_cache(cache_dir)

        rc = run_cache_clear(_make_args(cache_dir=cache_dir))

        assert rc == 0
        for f in seeded:
            assert not f.exists()
        # Empty pyiwfm_cache/ folder is pruned afterwards.
        assert not cache_dir.exists()
        out = capsys.readouterr().out
        assert f"Removed {len(seeded)} file(s)" in out

    def test_no_caches_found_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = run_cache_clear(_make_args(model_dir=tmp_path))
        assert rc == 0
        assert "No pyiwfm cache files found" in capsys.readouterr().out

    def test_model_dir_clears_cache_and_legacy_orphans(self, tmp_path: Path) -> None:
        results = tmp_path / "Results"
        cache_dir = results / "pyiwfm_cache"
        cache_files = _seed_cache(cache_dir)
        legacy = results / "model_cache.db"
        legacy.write_bytes(b"old")

        rc = run_cache_clear(_make_args(model_dir=tmp_path))

        assert rc == 0
        for f in cache_files:
            assert not f.exists()
        assert not legacy.exists()
        # The Results/ directory itself stays — only the empty pyiwfm_cache/
        # subfolder gets pruned.
        assert results.exists()
        assert not cache_dir.exists()


class TestParserRegistration:
    def test_cache_clear_invocation(self, tmp_path: Path) -> None:
        """Smoke test: argparse round-trips the documented CLI."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        add_cache_parser(sub)

        args = parser.parse_args(["cache", "clear", "--cache-dir", str(tmp_path), "--dry-run"])
        assert args.cache_command == "clear"
        assert args.cache_dir == tmp_path
        assert args.dry_run is True
        # Function dispatch is wired up.
        assert args.func is run_cache_clear
