#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys

MODULES = [
    "numpy", "scipy", "matplotlib", "yaml", "tensorflow",
    "sionna", "sionna.phy", "sionna.phy.channel", "sionna.phy.channel.tr38901",
    "sionna.phy.ofdm", "sionna.sys", "pandas", "torch",
]


def main() -> None:
    print("python executable:", sys.executable)
    print("python version   :", sys.version.replace("\n", " "))
    print("platform         :", platform.platform())
    for name in MODULES:
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", None)
            print(f"{name:30s}: imported" + (f", version={ver}" if ver else ""))
        except Exception as e:
            print(f"{name:30s}: IMPORT FAILED: {type(e).__name__}({e!r})")


if __name__ == "__main__":
    main()
