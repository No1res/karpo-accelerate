"""Build the published SM80 CUDA operators against the active PyTorch environment."""

import os
import shutil
import tarfile
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

CUTLASS_URL = "https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.2.1"
CUTLASS_SHA256 = "a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df"
VLLM_URL = "https://raw.githubusercontent.com/vllm-project/vllm/v0.19.1/"
HEADERS = {
    "csrc/core/math.hpp": "b80a3ffbbe5944c865331da95261696e430981b623d447adf24f272241104f59",
    "csrc/cutlass_extensions/common.hpp": "185491103f50fa2d568bbfbd801efce54474c36bcb0f92fe846badf468051181",
    "csrc/cutlass_extensions/epilogue/broadcast_load_epilogue_c2x.hpp":
        "b43a987382f027f3bdba3ac840c8f551c6215b1ee2a6d879c4c8f7c52518249a",
    "csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp":
        "90d98aedc34de86428d825e33287fe75a141f08431a61f32e6ee10b975b49326",
    "csrc/quantization/w8a8/cutlass/scaled_mm_c2x.cuh":
        "52d5094425ae642eed2f14b64d8c8084207452762b84599bb064e9d7f439f9aa",
    "csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm80_dispatch.cuh":
        "9cde6fda28687ead7dd850ebd9cfceec64ba442b3895c500114df6b4f1d72c0a",
}
OPERATORS = {
    "int8": ("karpo_ampere_cutlass", "karpo_ampere"),
    "int4": ("karpo_int4_fused_sm80", "karpo_ampere_int4_fused"),
}


def required_kernels(profile):
    return ("int8", "int4") if profile == "w4a4" else ("int8",)


def library_path(directory, kernel):
    extension, _ = OPERATORS[kernel]
    return Path(directory) / kernel / f"{extension}.so"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def download(path, url, expected):
    if path.is_file() and digest(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(url, timeout=30) as response:
                shutil.copyfileobj(response, stream)
            stream.flush()
            if digest(temporary) != expected:
                raise ValueError(f"Checksum mismatch for {url}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def include_paths(build, cutlass, vllm):
    if cutlass is None:
        archive = build / "sources" / "cutlass-4.2.1.tar.gz"
        download(archive, CUTLASS_URL, CUTLASS_SHA256)
        cutlass = archive.parent / "cutlass-4.2.1" / "include"
        if not (cutlass / "cutlass/cutlass.h").is_file():
            with tarfile.open(archive, "r:gz") as stream:
                headers = (member for member in stream if member.name.startswith("cutlass-4.2.1/include/"))
                stream.extractall(archive.parent, members=headers, filter="data")
    if not (cutlass / "cutlass/cutlass.h").is_file():
        raise ValueError("--cutlass-include must point to the CUTLASS 4.2.1 include directory")

    if vllm is None:
        vllm = build / "sources" / "vllm-0.19.1"
        for header, expected in HEADERS.items():
            download(vllm / header, VLLM_URL + header, expected)
    for header, expected in HEADERS.items():
        if not (vllm / header).is_file() or digest(vllm / header) != expected:
            raise ValueError(f"Expected the vLLM 0.19.1 header {header} under {vllm}")
    return [str((vllm / "csrc").resolve()), str(cutlass.resolve())]


def build_kernels(args):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None or torch.version.cuda is None:
        raise ValueError("Install CUDA-enabled PyTorch and set CUDA_HOME to a CUDA toolkit with nvcc")
    build = args.kernel_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    includes = include_paths(build, args.cutlass_include, args.vllm_source)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", "2")
    for kernel in required_kernels(args.profile):
        extension, namespace = OPERATORS[kernel]
        directory = build / kernel
        directory.mkdir(exist_ok=True)
        library = load(
            extension, [str(Path(__file__).with_name("kernels") / f"{kernel}.cu")],
            extra_include_paths=includes,
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
            build_directory=str(directory), is_python_module=False, verbose=True,
        )
        if not hasattr(getattr(torch.ops, namespace), "scaled_mm"):
            raise RuntimeError(f"The expected operator is missing in {library}")
        print(library, flush=True)


def load_kernels(directory, profile):
    import torch

    for kernel in required_kernels(profile):
        path = library_path(directory, kernel).resolve(strict=True)
        torch.ops.load_library(str(path))
