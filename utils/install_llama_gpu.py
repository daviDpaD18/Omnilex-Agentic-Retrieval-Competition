#!/usr/bin/env python3
"""Install llama-cpp-python with GPU support.

Auto-detects CUDA version and installs the appropriate prebuilt wheel.
Supports CUDA 12.1-12.5. Falls back to CPU if no compatible CUDA found.

Usage:
    python scripts/install_llama_gpu.py
    python scripts/install_llama_gpu.py --cuda 12.1  # Force specific version
    python scripts/install_llama_gpu.py --cpu        # Force CPU version
"""

import argparse
import os
import re
import subprocess
import sys

SUPPORTED_CUDA = ["12.5", "12.4", "12.3", "12.2", "12.1"]
WHEEL_BASE_URL = "https://abetlen.github.io/llama-cpp-python/whl"

VCVARSALL_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    r"C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
]


def get_cuda_version() -> str | None:
    """Detect installed CUDA version from nvcc or nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def find_compatible_cuda(detected: str) -> str | None:
    """Find the best compatible CUDA wheel version."""
    major_minor = detected.split(".")
    if len(major_minor) < 2:
        return None

    major = int(major_minor[0])
    minor = int(major_minor[1])

    if major != 12:
        return None

    for cuda_ver in SUPPORTED_CUDA:
        ver_parts = cuda_ver.split(".")
        ver_minor = int(ver_parts[1])
        if ver_minor <= minor:
            return cuda_ver

    return None


def get_msvc_env() -> dict | None:
    """Source vcvarsall.bat and return the resulting environment variables."""
    for vcvarsall in VCVARSALL_CANDIDATES:
        if not os.path.exists(vcvarsall):
            continue
        print(f"Found MSVC at: {vcvarsall}")
        result = subprocess.run(
            f'"{vcvarsall}" x64 && set',
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        env = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
        return env
    return None


def install_llama_cpp(cuda_version: str | None = None, force_cpu: bool = False) -> bool:
    """Install llama-cpp-python with appropriate GPU/CPU support."""
    env = os.environ.copy()

    if force_cpu:
        print("Installing CPU version (forced)...")
        cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", "llama-cpp-python"]
    elif cuda_version:
        cuda_tag = f"cu{cuda_version.replace('.', '')}"
        wheel_url = f"{WHEEL_BASE_URL}/{cuda_tag}"
        cuda_root = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_version}"
        nvcc = os.path.join(cuda_root, "bin", "nvcc.exe")

        print(f"Installing GPU version for CUDA {cuda_version}...")

        msvc_env = get_msvc_env()
        if msvc_env:
            print("MSVC environment loaded.")
            env.update(msvc_env)
        else:
            print("Warning: could not find vcvarsall.bat — C compiler may not be found.")

        env["CUDA_PATH"] = cuda_root
        env["CUDA_HOME"] = cuda_root
        env["PATH"] = os.path.join(cuda_root, "bin") + ";" + env.get("PATH", "")

        cmake_args = f'-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER="{nvcc}" -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -G Ninja'
        env["CMAKE_ARGS"] = cmake_args
        env["FORCE_CMAKE"] = "1"
        print(f"CMAKE_ARGS={cmake_args}")

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "llama-cpp-python",
            "--extra-index-url",
            wheel_url,
        ]
    else:
        print("Installing CPU version (no compatible CUDA found)...")
        cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", "llama-cpp-python"]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Install llama-cpp-python with GPU support")
    parser.add_argument(
        "--cuda",
        type=str,
        help="Force specific CUDA version (e.g., 12.1, 12.4)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only installation",
    )
    args = parser.parse_args()

    if args.cpu:
        success = install_llama_cpp(force_cpu=True)
    elif args.cuda:
        if args.cuda not in SUPPORTED_CUDA:
            print(f"Warning: CUDA {args.cuda} may not have prebuilt wheels.")
            print(f"Supported versions: {', '.join(SUPPORTED_CUDA)}")
        success = install_llama_cpp(cuda_version=args.cuda)
    else:
        detected = get_cuda_version()
        if detected:
            print(f"Detected CUDA version: {detected}")
            compatible = find_compatible_cuda(detected)
            if compatible:
                print(f"Using compatible wheel for CUDA {compatible}")
                success = install_llama_cpp(cuda_version=compatible)
            else:
                print(f"No prebuilt wheel for CUDA {detected}.")
                print(f"Supported: {', '.join(SUPPORTED_CUDA)}")
                print("Falling back to CPU version.")
                success = install_llama_cpp(force_cpu=True)
        else:
            print("No CUDA installation detected.")
            success = install_llama_cpp(force_cpu=True)

    if success:
        print("\n✓ Installation complete!")
        try:
            from omnilex.llm import has_cuda_support

            if has_cuda_support():
                print("✓ GPU support enabled")
            else:
                print("→ Running in CPU mode")
        except ImportError:
            pass
    else:
        print("\n✗ Installation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
