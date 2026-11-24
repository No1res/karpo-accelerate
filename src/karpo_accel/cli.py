"""Command line entry points for Karpo serving, calibration, and validation."""

import argparse
import math
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def finite(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative number")
    return number


def positive_real(value):
    number = finite(value)
    if number == 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return number


def port(value):
    number = positive(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("expected a port in 1 to 65535")
    return number


def add_model_options(parser):
    parser.add_argument("--checkpoint", type=Path, required=True, help="local Karpo checkpoint directory")
    parser.add_argument("--profile", choices=("bf16", "w8a8", "w4a4"), default="bf16")
    parser.add_argument("--kernel-dir", type=Path, default=Path("build/kernels"))
    parser.add_argument("--compact-rubrics", type=Path, help="JSON mapping of complete instructions to previews or null")
    parser.add_argument("--max-length", type=positive, default=32768)
    parser.add_argument("--batch-requests", type=positive, default=8)
    parser.add_argument("--batch-tokens", type=positive, default=4096)
    parser.add_argument("--long-tokens", type=nonnegative, help="route longer requests to cooperative scheduling")


def add_replicas(parser):
    parser.add_argument("--replica", action="append", required=True, help="replica origin; repeat for additional GPUs")
    parser.add_argument("--timeout", type=positive_real, default=60)


def arguments(argv=None):
    parser = argparse.ArgumentParser(prog="karpo-accelerate", description="Serve and benchmark Karpo decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="start one Karpo replica on the selected CUDA device")
    add_model_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=port, default=18440)
    serve.add_argument("--batch-wait-ms", type=finite, default=3)
    serve.add_argument("--graph-tokens", type=nonnegative, help="largest CUDA graph bucket; 0 disables graphs")
    serve.add_argument("--graph-padding", type=nonnegative, default=127)
    serve.add_argument("--exit-heads", type=Path, help="artifact produced by calibrate")
    serve.add_argument("--full-response", action="store_true", help="return the original response with option scores")

    build = commands.add_parser("build-kernels", help="compile the SM80 CUTLASS extensions")
    build.add_argument("--profile", choices=("w8a8", "w4a4"), default="w4a4")
    build.add_argument("--kernel-dir", type=Path, default=Path("build/kernels"))
    build.add_argument("--cutlass-include", type=Path)
    build.add_argument("--vllm-source", type=Path)

    calibration = commands.add_parser("calibrate", help="fit task-specific early exits on three disjoint JSONL splits")
    add_model_options(calibration)
    for name in ("fit", "calibration", "holdout", "output"):
        calibration.add_argument(f"--{name}", type=Path, required=True)
    calibration.add_argument("--depths", nargs="+", type=positive, default=[34, 42])
    calibration.add_argument("--margin", type=finite, default=0.02)
    calibration.add_argument("--priority-option", action="append", default=[],
                             help="calibrate the named option against false positives on labelled data")

    proxy = commands.add_parser("proxy", help="expose independent replicas through a single local HTTP endpoint")
    add_replicas(proxy)
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=port, default=18440)
    proxy.add_argument("--max-body-bytes", type=positive, default=32 * 1024 * 1024)

    bench = commands.add_parser("bench", help="measure open-loop throughput, P95, and accuracy")
    add_replicas(bench)
    bench.add_argument("--requests", type=Path, required=True)
    bench.add_argument("--qps", type=positive_real, required=True)
    bench.add_argument("--seconds", type=positive_real, default=180)
    bench.add_argument("--arrivals", choices=("uniform", "poisson"), default="uniform")
    bench.add_argument("--seed", type=int, default=20261121)
    bench.add_argument("--workers", type=positive, default=128)
    bench.add_argument("--deadline-ms", type=positive_real, default=350)
    bench.add_argument("--repeat", action="store_true", help="allow repeated contexts in a smoke test")
    bench.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command in ("serve", "calibrate"):
        if not (args.checkpoint / "decision_config.json").is_file():
            parser.error("--checkpoint must contain decision_config.json")
        if args.long_tokens is None:
            args.long_tokens = 0 if args.profile == "bf16" else 1536
    if args.command == "serve":
        if args.graph_tokens is None:
            args.graph_tokens = 0 if args.profile == "bf16" else 4096
        if args.graph_tokens and not 224 <= args.graph_tokens <= 4096:
            parser.error("--graph-tokens must be 0 or in 224 to 4096")
        if args.exit_heads is not None and (not args.graph_tokens or args.full_response):
            parser.error("--exit-heads requires CUDA graphs and a choice-only response")
    if args.command == "calibrate":
        args.graph_tokens = 0
        if len(args.depths) > 2 or args.depths != sorted(set(args.depths)):
            parser.error("--depths must specify one or two distinct layers in ascending order")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        match args.command:
            case "serve":
                from .runtime import serve
                serve(args)
            case "calibrate":
                from .calibrate import calibrate
                calibrate(args)
            case "build-kernels":
                from .build_kernels import build_kernels
                build_kernels(args)
            case "proxy":
                from .proxy import run_proxy
                run_proxy(args)
            case "bench":
                from .benchmark import run_benchmark
                run_benchmark(args)
    except ModuleNotFoundError as error:
        if error.name in ("torch", "triton", "karpo_decision"):
            raise SystemExit("Install the CUDA dependencies described in README.md") from error
        raise
    except (OSError, ValueError) as error:
        raise SystemExit(f"karpo-accelerate: {error}") from error
