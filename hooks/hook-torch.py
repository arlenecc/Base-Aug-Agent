# Custom torch hook for PyInstaller.
#
# Problem: PyInstaller's stock hook-torch.py calls collect_submodules("torch")
# inside an isolated subprocess, which aborts with SIGABRT (exit -6) on this
# macOS / PyTorch combo while scanning torch.distributed. That kills the build.
#
# Fix: enumerate torch's submodule *names* by walking the package directory on
# disk (no imports -> no abort), and still bundle torch's data files and
# dynamic libs so the package works at runtime. Heavy / macOS-irrelevant
# submodules are excluded to keep the bundle lean.
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


def _torch_submodule_names():
    import torch

    base = torch.__path__[0]
    names = ["torch"]
    for root, dirs, files in os.walk(base):
        rel = os.path.relpath(root, base)
        if rel == ".":
            for f in files:
                if f.endswith(".py") and f != "__init__.py":
                    names.append("torch." + f[:-3])
            for d in dirs:
                if os.path.isfile(os.path.join(root, d, "__init__.py")):
                    names.append("torch." + d)
            continue
        prefix = "torch." + rel.replace(os.sep, ".")
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                names.append(prefix + "." + f[:-3])
        for d in dirs:
            if os.path.isfile(os.path.join(root, d, "__init__.py")):
                names.append(prefix + "." + d)
    return names


hiddenimports = _torch_submodule_names()
datas = collect_data_files("torch")
binaries = collect_dynamic_libs("torch")
excludedimports = [
    "torch.distributed",
    "torch.testing",
    "torch.jit",
    "torch._dynamo",
    "torch.compile",
    "torch.benchmark",
    "torch.utils.tensorboard",
    "torch.profiler",
]
