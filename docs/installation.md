# Installation

Run commands from the repository root. Python 3.12 is the tested baseline.
PolyBot is a repository-run research project; the CLI scripts bootstrap `src/`
and the pytest configuration provides the test import path.

## Native prerequisites

[PolyEnv installation](https://polyenv.readthedocs.io/en/latest/installation.html)
requires Git, CMake 3.20+ and a C++20-capable compiler when building its Python
extension. PolyBot pins the tested engine revision in `requirements.txt`.

| Platform | Compiler setup |
| --- | --- |
| Windows | Visual Studio Build Tools with Desktop development with C++; use a Developer PowerShell when the compiler is not detected |
| Linux | A C++20 compiler and CMake; install the corresponding distribution development packages |
| macOS | Xcode Command Line Tools and CMake |

Use the same Python interpreter for `pip`, tests and training. Native build
failures must be resolved before increasing worker counts or starting a long run.
Repository text files use LF endings on all platforms so recorded source hashes
remain stable after a Windows checkout.

## Environment

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate in Windows PowerShell:

```powershell
.venv/Scripts/Activate.ps1
```

An existing Conda environment is also suitable. Install PyTorch for the intended
device using the [official selector](https://pytorch.org/get-started/locally/)
before installing this project's dependencies, especially for CUDA.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/check_environment.py --smoke
python -m pytest
```

For training without development tools, use `requirements.txt` instead.
The dependency ranges are compatibility bounds, not a fully resolved lockfile.
Record the actual environment for each experiment.

## Devices

`--device cuda`, `--device mps` and `--device cpu` select explicit devices.
`--device auto` uses the available backends through the project's device
selector. CUDA is the principal throughput target; CPU is useful for tests
and small experiments. MPS operation support and performance depend on the
installed PyTorch build. A native C++ engine and a GPU network are separate:
selecting CUDA does not move game simulation to the GPU.

`check_environment.py` reports package versions, the installed PolyEnv VCS
revision when available, device availability and required native methods.
It deliberately omits local installation paths. A successful reset checks
the public `[1, 121, 23]` map packet and legal actions for an 11×11 map.

## Updating PolyEnv

The pinned revision is a reproducible API baseline. To test a newer engine,
install its chosen revision explicitly, run the environment check and the full
tests, then update the pin in a reviewed change. Merely pulling an engine
checkout does not rebuild its installed C++ extension.

## Common problems

- **Import fails:** compare `python -m pip --version` with the interpreter
  used to run the script; activate the intended environment.
- **A native method or buffer signature is missing:** reinstall the pinned
  PolyEnv revision in that environment and rerun the smoke check.
- **CUDA is unavailable:** verify the PyTorch build and driver using the
  official installation instructions; the engine package does not enable CUDA.
- **Memory pressure:** start with fewer environments, a smaller replay and
  fewer pending MCTS leaves. Model weights are only part of total memory use.

Next: [training](training.md), [architecture](architecture.md).
