"""Custom hatch build hook to build the React frontend before packaging."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build the React frontend during ``hatch build``."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:  # noqa: ARG002
        frontend_dir = Path(self.root) / "frontend"
        static_dir = Path(self.root) / "src" / "pyiwfm" / "visualization" / "webapi" / "static"

        # Skip if frontend source doesn't exist (e.g. installing from sdist without it)
        if not frontend_dir.exists():
            self._log("frontend/ directory not found, skipping frontend build")
            return

        # Check for Node.js
        npm = shutil.which("npm")
        if npm is None:
            if (static_dir / "index.html").exists():
                self._log("npm not found; using existing pre-built frontend assets")
                return
            self._log("WARNING: npm not found and no pre-built frontend assets exist")
            return

        self._log("Building frontend...")
        try:
            subprocess.run(
                [npm, "ci"],
                cwd=str(frontend_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [npm, "run", "build"],
                cwd=str(frontend_dir),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            if (static_dir / "index.html").exists():
                self._log(f"Frontend build failed ({exc}); using existing pre-built assets")
                return
            raise RuntimeError(
                f"Frontend build failed and no pre-built assets exist:\n{exc.stderr}"
            ) from exc

        if not (static_dir / "index.html").exists():
            raise RuntimeError(
                "Frontend build completed but static/index.html not found. "
                "Check that vite.config.ts outputs to the correct directory."
            )

        self._log("Frontend build complete")

    def _log(self, msg: str) -> None:
        self.app.display_info(f"[pyiwfm] {msg}")
