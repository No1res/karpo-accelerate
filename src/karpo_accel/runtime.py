"""Connect the acceleration components to the pinned Karpo HTTP protocol."""

import json
import signal
from hashlib import sha256
from http.server import ThreadingHTTPServer
from threading import Event, Thread

import torch
from karpo_decision.serve import server

from .batch import BatchRunner, TopChoiceHandler
from .build_kernels import load_kernels
from .rubric import SingleRubricLayout

UPSTREAM_REVISION = "6600011eb5cf46232d056ec0717dc726146d29de"


def read_rubrics(path):
    if path is None:
        return {}
    rubrics = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rubrics, dict) or any(
        not key or (value is not None and (not isinstance(value, str) or not value))
        for key, value in rubrics.items()
    ):
        raise ValueError("Compact rubrics must map each complete instruction to a short preview or null")
    return rubrics


def checkpoint_fingerprint(path):
    """Bind a head to the model metadata and weight shard layout without reading all tensor data."""
    digest = sha256()
    for name in ("config.json", "decision_config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json", "readout.safetensors"):
        source = path / name
        if source.is_file():
            digest.update(name.encode())
            digest.update(sha256(source.read_bytes()).digest())
    for source in sorted(path.glob("model*.safetensors")):
        digest.update(source.name.encode())
        digest.update(str(source.stat().st_size).encode())
        with source.open("rb") as stream:
            digest.update(stream.read(65536))
            stream.seek(max(0, source.stat().st_size - 65536))
            digest.update(stream.read())
    return digest.hexdigest()


def identity(args, rubrics):
    return {"code_revision": UPSTREAM_REVISION, "checkpoint_fingerprint": checkpoint_fingerprint(args.checkpoint),
            "profile": args.profile, "rubrics": rubrics, "max_length": args.max_length,
            "long_tokens": args.long_tokens, "batch_tokens": args.batch_tokens, "batch_requests": args.batch_requests}


@torch.inference_mode()
def load_model(args):
    if not torch.cuda.is_available():
        raise ValueError("Serving and feature capture require a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    if (args.profile != "bf16" or args.long_tokens) and torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("The Ampere quantization and long-context kernels require SM80; use bf16 with --long-tokens 0")
    rubrics = read_rubrics(args.compact_rubrics)
    if args.profile != "bf16":
        load_kernels(args.kernel_dir, args.profile)
    decider, warmup = server.load_decider(str(args.checkpoint), device, args.max_length)
    if (args.profile != "bf16" or args.long_tokens or args.graph_tokens) and not getattr(decider.model, "prune_last", False):
        raise ValueError("This acceleration path requires Karpo's marker-pruned Gemma backbone")
    if rubrics:
        layout = decider.encoder.blocks
        if layout is None:
            raise ValueError("Compact rubrics require a Gemma block layout")
        compact = {text for text, preview in rubrics.items() if preview is None}
        previews = {text: preview for text, preview in rubrics.items() if preview is not None}
        decider.encoder.blocks = SingleRubricLayout(decider.encoder, layout.template, compact, previews)
    if args.profile != "bf16":
        from .projections import install_projections

        install_projections(decider.model, args.profile)
    decider.decide(warmup)
    return decider, rubrics


class HTTPServer(server.Server):
    def __init__(self, address, ranks, name, info, handler):
        self.ranks, self.name, self.info = ranks, name, info
        ThreadingHTTPServer.__init__(self, address, handler)


def prepare_graphs(args, decider, rubrics):
    from .graphs import SIZES, CascadeGraphs, ExitGraphs, PackedGraphs

    sizes = tuple(size for size in SIZES if size <= args.graph_tokens)
    config = (decider.model, sizes, args.batch_requests, args.graph_padding)
    policies = []
    if args.exit_heads is None:
        graphs = PackedGraphs(*config, shared_pool=True)
    else:
        from .exits import ExitPolicy

        data = torch.load(args.exit_heads, map_location="cpu", weights_only=True)
        if data.get("format") != 1 or data.get("runtime") != identity(args, rubrics):
            raise ValueError("Exit heads require the same checkpoint, precision, and rubric options as calibration")
        stages = data["stages"]
        if not 1 <= len(stages) <= 2:
            raise ValueError("An exit artifact must contain one or two stages")
        depth = stages[-1]["depth"]
        if len(stages) == 2:
            graphs = CascadeGraphs(*config, depth=depth, first_depth=stages[0]["depth"], shared_pool=True)
        else:
            graphs = ExitGraphs(*config, depth=depth, shared_pool=True)
        width = decider.readout.weight.shape[1]
        policies = [ExitPolicy(stage["heads"], width, decider.device) for stage in stages]
    with torch.cuda.device(decider.device):
        graphs.capture()
        graphs.install()
    return graphs, policies


def serve(args):
    decider, rubrics = load_model(args)
    options = {"max_requests": args.batch_requests, "max_tokens": args.batch_tokens, "wait_ms": args.batch_wait_ms}
    graphs, policies = prepare_graphs(args, decider, rubrics) if args.graph_tokens else (None, [])
    if policies:
        from .exits import ExitRunner

        runner = ExitRunner(decider, decider.device, graphs=graphs, policies=policies,
                            long_tokens=args.long_tokens or args.max_length, **options)
    elif args.long_tokens:
        from .long_context import ChunkedRunner

        runner = ChunkedRunner(decider, decider.device, long_tokens=args.long_tokens, **options)
    else:
        runner = BatchRunner(decider, decider.device, **options)
    depths = [graphs.depth] if policies else []
    if len(policies) == 2:
        depths.insert(0, graphs.first_depth)
    info = {"profile": args.profile, "graph_tokens": args.graph_tokens, "long_tokens": args.long_tokens,
            "early_exit_depths": depths}
    handler = server.Handler if args.full_response else TopChoiceHandler
    with HTTPServer((args.host, args.port), server.Ranks(runner, []), "karpo-decision", info, handler) as httpd:
        stop = Event()
        prior = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        worker = Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        print(json.dumps({"event": "listening", "host": args.host, "port": args.port, **info}), flush=True)
        try:
            stop.wait()
        finally:
            httpd.shutdown()
            worker.join()
            for sig, previous in prior.items():
                signal.signal(sig, previous)
