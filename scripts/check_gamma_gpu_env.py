"""Check whether this Python environment can run the v2.8 CuPy Gamma backend.

Usage:
    python scripts/check_gamma_gpu_env.py

The script is read-only. It does not install packages or change GPU settings.
Exit code 0 means the CuPy Gamma smoke test passed; exit code 1 means that the
GPU backend is not ready. CPU-only project dependencies are reported separately.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from typing import Any


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_check(module: str, distribution: str | None = None) -> tuple[bool, Any]:
    package = distribution or module
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", None) or _version(package) or "unknown"
        print(f"[PASS] {package}: version={version}")
        return True, imported
    except Exception as exc:
        print(f"[FAIL] {package}: {type(exc).__name__}: {exc}")
        return False, None


def _cuda_version_text(value: int) -> str:
    # CUDA integer version convention: 12060 -> 12.6, 13000 -> 13.0.
    major = int(value) // 1000
    minor = (int(value) % 1000) // 10
    return f"{major}.{minor} (raw={value})"


def main() -> int:
    print("=== Python ===")
    print(f"executable : {sys.executable}")
    print(f"version    : {sys.version.split()[0]}")
    print(f"platform   : {platform.platform()}")

    print("\n=== Project base packages ===")
    base_ok = True
    for module, distribution in (
        ("numpy", "numpy"),
        ("yaml", "PyYAML"),
        ("matplotlib", "matplotlib"),
    ):
        ok, _ = _import_check(module, distribution)
        base_ok = base_ok and ok

    print("\n=== NVIDIA system commands ===")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            print(f"[PASS] nvidia-smi: {nvidia_smi}")
            for line in result.stdout.splitlines():
                print(f"[INFO] GPU: {line}")
        except Exception as exc:
            print(f"[FAIL] nvidia-smi execution: {type(exc).__name__}: {exc}")
    else:
        print("[FAIL] nvidia-smi not found in PATH")

    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run(
                [nvcc, "--version"], check=True, capture_output=True, text=True, timeout=15
            )
            last_line = result.stdout.strip().splitlines()[-1]
            print(f"[INFO] CUDA Toolkit compiler: {last_line}")
        except Exception as exc:
            print(f"[WARN] nvcc query failed: {type(exc).__name__}: {exc}")
    else:
        print("[INFO] nvcc not found; this is allowed when wheel/conda CUDA components are used")

    print("\n=== Installed CuPy distributions ===")
    installed_cupy = {
        name: version
        for name in ("cupy", "cupy-cuda12x", "cupy-cuda13x")
        if (version := _version(name)) is not None
    }
    if installed_cupy:
        for name, version in installed_cupy.items():
            print(f"[INFO] {name}=={version}")
    else:
        print("[FAIL] No CuPy distribution found")
    if len(installed_cupy) > 1:
        print("[FAIL] Multiple CuPy distributions are installed; keep exactly one")

    print("\n=== CuPy / CUDA / GPU ===")
    cupy_ok, cp = _import_check("cupy", next(iter(installed_cupy), "cupy"))
    gpu_ok = False
    if cupy_ok:
        try:
            runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
            driver_version = int(cp.cuda.runtime.driverGetVersion())
            device_count = int(cp.cuda.runtime.getDeviceCount())
            print(f"[PASS] CUDA runtime: {_cuda_version_text(runtime_version)}")
            print(f"[PASS] NVIDIA driver API: {_cuda_version_text(driver_version)}")
            print(f"[PASS] visible CUDA devices: {device_count}")
            if device_count < 1:
                raise RuntimeError("CuPy imported, but no CUDA device is visible")

            for device_id in range(device_count):
                with cp.cuda.Device(device_id):
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    raw_name = props.get("name", b"unknown")
                    name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
                    free_b, total_b = cp.cuda.runtime.memGetInfo()
                    capability = f"{props.get('major', '?')}.{props.get('minor', '?')}"
                    print(
                        f"[PASS] GPU {device_id}: {name}; compute capability={capability}; "
                        f"memory free/total={free_b / 2**30:.2f}/{total_b / 2**30:.2f} GiB"
                    )

            # Exercise the operations and dtype used by measurement.py:
            # complex128 batched einsum, RX projection, gather, and BxB Gamma.
            with cp.cuda.Device(0):
                batch, freq, nr, nt, beams, rx_beams = 2, 3, 2, 4, 5, 3
                h = cp.ones((batch, freq, nr, nt), dtype=cp.complex128)
                w = cp.ones((beams, nt), dtype=cp.complex128)
                q = cp.ones((rx_beams, nr), dtype=cp.complex128)
                hw = cp.einsum("ufrt,kt->ukfr", h, w, optimize=True)
                z = cp.einsum("rn,ubfn->urbf", cp.conj(q), hw, optimize=True)
                power = cp.mean(cp.abs(z) ** 2, axis=3)
                selected = cp.argmax(power, axis=1)
                u_index = cp.arange(batch)[:, None]
                b_index = cp.arange(beams)[None, :]
                service = power[u_index, selected, b_index]
                interference = power[u_index, selected, :]
                gamma = service[:, :, None] / cp.maximum(interference + 0.1, 1e-30)
                cp.cuda.Stream.null.synchronize()
                if gamma.shape != (batch, beams, beams) or not bool(cp.all(cp.isfinite(gamma))):
                    raise RuntimeError(f"unexpected Gamma smoke-test result: shape={gamma.shape}")
                print(
                    f"[PASS] Gamma CUDA smoke test: shape={gamma.shape}, "
                    f"dtype={gamma.dtype}, sample={float(gamma[0, 0, 0].get()):.6g}"
                )
                gpu_ok = True
        except Exception as exc:
            print(f"[FAIL] CUDA Gamma smoke test: {type(exc).__name__}: {exc}")

    print("\n=== Existing optional Sionna/TensorFlow stack ===")
    print("These are not new CuPy requirements, but your configured channel/link backend may need them.")
    _import_check("sionna", "sionna")
    tf_ok, tf = _import_check("tensorflow", "tensorflow")
    if tf_ok:
        try:
            tf_gpus = tf.config.list_physical_devices("GPU")
            print(f"[INFO] TensorFlow visible GPUs: {len(tf_gpus)} {tf_gpus}")
        except Exception as exc:
            print(f"[WARN] TensorFlow GPU query failed: {type(exc).__name__}: {exc}")

    print("\n=== Result ===")
    if not base_ok:
        print("[WARN] Some base project packages are missing.")
    if gpu_ok and len(installed_cupy) == 1:
        print("[READY] v2.8 measurement.gamma_backend=cupy can be used in this environment.")
        return 0
    print("[NOT READY] The CuPy Gamma backend is not usable yet.")
    if not installed_cupy:
        print('CUDA 12.x: python -m pip install ".[gpu]"')
        print('CUDA 13.x: python -m pip install ".[gpu-cuda13]"')
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
