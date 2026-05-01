"""Enable running pyiwfm as a module: python -m pyiwfm."""

from __future__ import annotations

import sys

from pyiwfm.cli import main

sys.exit(main())
