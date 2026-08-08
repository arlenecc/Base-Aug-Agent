"""Entry point: launch the PyQt6 base-agent UI."""
from __future__ import annotations

import os
import sys


def main() -> None:
    # When running from source, add src/ to the path so the `agent` package
    # is importable.  When running as a PyInstaller-frozen binary, the
    # package is already bundled and sys.path is set up automatically.
    if not getattr(sys, "frozen", False):
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(here, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)

    from agent.ui.main_window import run

    run()


if __name__ == "__main__":
    main()
