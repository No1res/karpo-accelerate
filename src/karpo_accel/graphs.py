"""Capture token buckets while isolating requests and padding sequences."""

from bisect import bisect_left
from collections import Counter

import torch
from karpo_decision.families.gemma.packed import PaddedPlan, block_plan
from karpo_decision.model.packed import Pack
from karpo_decision.serve import server

SIZES = (224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 640, 704, 768, 832, 896, 960, 1024,
         *range(1088, 2049, 64), *range(2176, 4097, 128))


class PackedGraphs:
    def __init__(self, model, sizes, maximum_requests, padding, span_limit=None, shared_pool=False):
        self.model, self.sizes, self.maximum_requests, self.padding = model, sizes, maximum_requests, padding
        self.span_limit = span_limit
        self.shared_pool = shared_pool
        self.weight = model.lm.embed_tokens.weight
        self.device = self.weight.device
        self.ids = torch.zeros(sizes[-1], dtype=torch.long, device=self.device)
        self.pos = self.ids.clone()
        self.cu = torch.zeros(maximum_requests + 2, dtype=torch.int32, device=self.device)
        self.out = self.weight.new_zeros((1, sizes[-1], self.weight.shape[1]))
        self.graphs = {}

    def padded_pack(self, size, span):
        return Pack(input_ids=self.ids[:size], seqpos=self.pos[:size], cu=self.cu, cu_cpu=None,
                    max_length=span, markers=self.ids[:0], marker_cu=self.cu[:1], max_markers=0,
                    plan=PaddedPlan(self.cu, span))

    def body(self, size, span):
        p = self.padded_pack(size, span)
        h = self.model.embed(p.input_ids)
        self.out[:, :size].copy_(self.model.body(h, self.model.rope(h, p.seqpos), p))

    def capture_stages(self):
        return ((self.body, self.graphs),)

    @torch.inference_mode()
    def capture(self):
        side = torch.cuda.Stream(device=self.device)
        active = torch.cuda.current_stream(self.device)
        for forward, graphs in self.capture_stages():
            pool = torch.cuda.graph_pool_handle() if self.shared_pool else None
            for size in self.sizes:
                spans = [size]
                if self.span_limit is not None and self.span_limit < size <= self.span_limit * (self.maximum_requests + 1):
                    spans.append(self.span_limit)
                for span in spans:
                    bounds = list(range(0, size, span))
                    bounds.extend([size] * (self.cu.numel() - len(bounds)))
                    self.cu.copy_(torch.tensor(bounds, dtype=torch.int32))
                    side.wait_stream(active)
                    with torch.cuda.stream(side):
                        forward(size, span)
                        forward(size, span)
                    active.wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        forward(size, span)
                    graphs[size, span] = graph

    def bucket(self, p):
        index = bisect_left(self.sizes, p.tokens)
        if index == len(self.sizes):
            return "over_budget", None
        size = self.sizes[index]
        if size - p.tokens > self.padding:
            return "padding", None
        if p.pixel_values is not None or p.cu_cpu is None or len(p.cu_cpu) - 1 > self.maximum_requests:
            return "layout", None
        if p.segment_ids_cpu is None or bool(p.segment_ids_cpu.any()):
            return "segments", None
        span = self.span_limit if p.max_length <= (self.span_limit or 0) else size
        return "replay", (size, span) if (size, span) in self.graphs else (size, size)

    def load_pack(self, p, size):
        self.ids[:p.tokens].copy_(p.input_ids)
        self.pos[:p.tokens].copy_(p.seqpos)
        # The isolated pad sequence and any empty entries all end at the bucket size.
        self.cu.fill_(size)
        self.cu[:p.cu.numel()].copy_(p.cu)

    def plan_head(self, p):
        p.plan = block_plan(p, self.model.window)

    def head(self, p, hidden):
        layer_type = self.model.types[-1]
        rope = {layer_type: self.model.lm.rotary_emb(hidden, p.seqpos[None], layer_type)}
        return self.model.head(hidden, rope, p)

    def replay(self, p, key):
        self.load_pack(p, key[0])
        self.plan_head(p)
        self.graphs[key].replay()
        return self.head(p, self.out[:, :p.tokens])

    def install(self):
        original = self.model.forward
        counts = Counter()
        calls = 0

        def replay(p):
            nonlocal calls
            path, key = self.bucket(p)
            calls += 1
            counts[path] += 1
            if calls % 512 == 0:
                server.log(event="packed_graph_usage", **counts)
            return original(p) if key is None else self.replay(p, key)

        self.model.forward = replay


class ExitGraphs(PackedGraphs):
    def __init__(self, *args, depth, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.last = len(self.model.layers) - 1
        if not (self.model.prune_last and 0 < depth < self.last):
            raise ValueError("An exit must be within the body of a marker-pruned model")
        self.before = torch.zeros_like(self.out)
        self.tail_input = torch.zeros_like(self.out)
        self.tail_graphs = {}

    def layers(self, h, p, start, stop):
        rope = self.model.rope(h, p.seqpos)
        for i in range(start, stop):
            h = self.model.run(i, h, *rope[self.model.types[i]], p)
        return h

    def prefix(self, size, span):
        p = self.padded_pack(size, span)
        self.before[:, :size].copy_(self.layers(self.model.embed(p.input_ids), p, 0, self.depth))

    def suffix(self, size, span):
        p = self.padded_pack(size, span)
        self.out[:, :size].copy_(self.layers(self.tail_input[:, :size], p, self.depth, self.last))

    def capture_stages(self):
        return ((self.prefix, self.graphs), (self.suffix, self.tail_graphs))

    def start(self, p, key):
        self.load_pack(p, key[0])
        self.graphs[key].replay()
        return self.before[:, :key[0]]

    def load_tail(self, p, key, source, old_bounds, kept):
        if old_bounds is None:
            self.tail_input[:, :key[0]].copy_(source)
        else:
            offset = 0
            for i in kept:
                start, stop = old_bounds[i], old_bounds[i + 1]
                self.tail_input[:, offset:offset + stop - start].copy_(source[:, start:stop])
                offset += stop - start
            self.tail_input[:, offset:key[0]].zero_()
            self.load_pack(p, key[0])

    def finish(self, p, key, source, old_bounds=None, kept=None):
        self.load_tail(p, key, source, old_bounds, kept)
        self.plan_head(p)
        self.tail_graphs[key].replay()
        return self.head(p, self.out[:, :p.tokens])

    def replay(self, p, key):
        return self.finish(p, key, self.start(p, key))


class CascadeGraphs(ExitGraphs):
    def __init__(self, *args, first_depth, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < first_depth < self.depth:
            raise ValueError("The first exit must precede the incumbent exit")
        self.first_depth = first_depth
        self.middle_out = torch.zeros_like(self.out)
        self.middle_graphs = {}

    def prefix(self, size, span):
        p = self.padded_pack(size, span)
        self.before[:, :size].copy_(self.layers(self.model.embed(p.input_ids), p, 0, self.first_depth))

    def middle(self, size, span):
        p = self.padded_pack(size, span)
        hidden = self.layers(self.tail_input[:, :size], p, self.first_depth, self.depth)
        self.middle_out[:, :size].copy_(hidden)

    def capture_stages(self):
        return ((self.prefix, self.graphs), (self.middle, self.middle_graphs), (self.suffix, self.tail_graphs))

    def start_first(self, p, key):
        return super().start(p, key)

    def advance(self, p, key, source, old_bounds=None, kept=None):
        self.load_tail(p, key, source, old_bounds, kept)
        self.middle_graphs[key].replay()
        return self.middle_out[:, :key[0]]

    def start(self, p, key):
        # Callers bypassing the new exit still receive the incumbent layer's features.
        return self.advance(p, key, self.start_first(p, key))
