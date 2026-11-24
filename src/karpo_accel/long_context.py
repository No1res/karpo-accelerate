"""Interleave short batches with long-context layers and attention blocks."""

from collections import deque
from functools import lru_cache
from queue import Empty
from time import perf_counter

import torch
from karpo_decision.answers import answer
from karpo_decision.families.gemma import fused, sliding_attention
from karpo_decision.families.gemma.packed import block_plan
from karpo_decision.model.packed import pack
from karpo_decision.model.readout import per_question
from karpo_decision.serve import server

from .batch import BatchRunner
from .global_attention import grouped
from .projections import project_attention


class PreemptiveRunner(BatchRunner):
    def __init__(self, decider, device, *, long_tokens, long_short_share=0.8, shorter_first_ms=80, **kwargs):
        if not getattr(decider.model, "prune_last", False):
            raise ValueError("Layer-boundary scheduling requires Karpo's eager bidirectional Gemma path")
        self.long_tokens = long_tokens
        self.long_short_share = long_short_share
        self.shorter_first_seconds = shorter_first_ms / 1000
        super().__init__(decider, device, **kwargs)

    def _long(self, job):
        return job.packable and job.tokens > self.long_tokens

    def _next_job(self, backlog):
        if not self.shorter_first_seconds:
            return backlog.popleft() if backlog else self.pending.get()
        if not backlog:
            backlog.append(self.pending.get())
        if perf_counter() - backlog[0].received >= self.shorter_first_seconds:
            return backlog.popleft()
        while len(backlog) < self.pending.maxsize:
            try:
                backlog.append(self.pending.get_nowait())
            except Empty:
                break
        chosen = min(range(len(backlog)), key=lambda index: backlog[index].tokens)
        first = backlog[chosen]
        del backlog[chosen]
        return first

    def _work(self):
        torch.cuda.set_device(self.device)
        backlog = deque()
        while True:
            first = self._next_job(backlog)
            if self._long(first):
                try:
                    response = self._run_long(first, backlog)
                except Exception as exc:
                    first.result.set_exception(exc)
                else:
                    first.result.set_result(response)
            else:
                self._run_batch(self._take_pack(first, backlog))

    def _take_pack(self, first, backlog):
        jobs, remaining = [first], self.max_tokens - first.tokens
        if not first.packable:
            return jobs
        deadline = first.received + self.wait_seconds
        while len(jobs) < self.max_requests and remaining > 0:
            try:
                job = backlog.popleft() if backlog else self.pending.get(timeout=max(0.0, deadline - perf_counter()))
            except Empty:
                break
            if self._long(job) or not job.packable or job.tokens > remaining:
                backlog.appendleft(job)
                break
            jobs.append(job)
            remaining -= job.tokens
        return jobs

    def _ready_short_pack(self, backlog):
        while len(backlog) < self.pending.maxsize:
            try:
                backlog.append(self.pending.get_nowait())
            except Empty:
                break
        jobs, remaining = [], self.max_tokens
        for _ in range(len(backlog)):
            job = backlog.popleft()
            if (job.packable and not self._long(job) and job.tokens <= remaining
                    and len(jobs) < self.max_requests):
                jobs.append(job)
                remaining -= job.tokens
            else:
                backlog.append(job)
        return jobs

    def _preemptible_layer(self, model, index, h, cos, sin, p, ready):
        h = model.run(index, h, cos, sin, p)
        ready()
        return h

    @torch.inference_mode()
    def _run_long(self, job, backlog):
        started = perf_counter()
        decider = self.decider
        model = decider.model
        encoded = job.plan[0]
        p = pack(job.plan, decider.ckpt.backbone, self.device)
        p.plan = block_plan(p, model.window)
        interleaved = 0
        short_seconds = 0.0
        stream = torch.cuda.current_stream(self.device)
        def ready():
            nonlocal interleaved, short_seconds
            stream.synchronize()
            now = perf_counter()
            if self.long_short_share < 1 and short_seconds > (now - started) * self.long_short_share:
                return
            batch = self._ready_short_pack(backlog)
            if batch:
                self._run_batch(batch)
                short_seconds += perf_counter() - now
                interleaved += len(batch)

        hidden = model.embed(p.input_ids)
        rope = model.rope(hidden, p.seqpos)
        for index, layer_type in enumerate(model.types[:-1]):
            hidden = self._preemptible_layer(model, index, hidden, *rope[layer_type], p, ready)

        flat = decider.readout(model.head(hidden, rope, p)).cpu()
        questions = job.request["questions"]
        answers = {}
        for key, logits in zip(encoded.keys, per_question(flat, encoded.counts), strict=True):
            question = questions[key]
            scaled = logits / decider.config.temperatures[question["type"]]
            answers[key] = answer(question, scaled.softmax(-1).tolist())
        server.log(event="preemptible_long", tokens=job.tokens, interleaved=interleaved,
                   interleaved_ms=round(1000 * short_seconds, 2),
                   elapsed_ms=round(1000 * (perf_counter() - started), 2))
        return {"answers": answers, "usage": {"input_tokens": job.tokens, "output_tokens": 0, "sequences": 1}}


class StripedRunner(PreemptiveRunner):
    stripe_rows = 1024
    minimum_tokens = 8192

    def _can_stripe_global(self, model, index, p):
        return (p.tokens >= self.minimum_tokens
                and model.types[index] == "full_attention" and len(p.plan.q_cu) == 2
                and p.plan.images is None and not model.causal_global)

    def _should_chunk_ffn(self, h):
        return False

    def _finish_ffn(self, model, layer, h, ready):
        h = fused.add_scale(h, model._mlp(layer, h), layer.layer_scalar)
        ready()
        return h

    def _preemptible_layer(self, model, index, h, cos, sin, p, ready):
        n = p.tokens
        if not self._can_stripe_global(model, index, p):
            return super()._preemptible_layer(model, index, h, cos, sin, p, ready)
        original = model._global
        def global_forward(q, k, v, qb, kb, lens, causal, scale):
            if q.shape[1] != n or k.shape[1] != n or len(qb) != 2 or len(kb) != 2 or causal:
                return original(q, k, v, qb, kb, lens, causal, scale)
            out = torch.empty_like(q[0], memory_format=torch.contiguous_format)
            for i in range(0, n, self.stripe_rows):
                queries = q[0, i:i + self.stripe_rows]
                piece = grouped(queries, k[0], v[0], scale, (64, 64, 8, 1))
                out[i:i + self.stripe_rows].copy_(piece)
                ready()
            return out.view(n, -1)

        layer = model.layers[index]
        model._global = global_forward
        try:
            attn = model.attention(layer.self_attn, model.norm_input(layer.input_layernorm, h), cos, sin, p)
        finally:
            model._global = original
        h = fused.residual_norm(h, layer.post_attention_layernorm, attn)
        ready()
        return self._finish_ffn(model, layer, h, ready)


@lru_cache(maxsize=16)
def sliding_segments(tokens, rows, left_window, device):
    ranges = [(start, min(tokens, start + rows), max(0, start - left_window))
              for start in range(0, tokens, rows)]
    boundaries = torch.tensor(
        [[0, count] for start, stop, halo in ranges for count in (stop - start, stop - halo)],
        device=device, dtype=torch.int32,
    ).unbind()
    return tuple((*bounds, q_cu, k_cu) for bounds, q_cu, k_cu in
                 zip(ranges, boundaries[::2], boundaries[1::2], strict=True))


class ChunkedRunner(StripedRunner):
    ffn_rows = 2048
    sliding_rows = 2048

    def _should_chunk_ffn(self, h):
        return 0 < self.ffn_rows < h.shape[1]

    def _finish_ffn(self, model, layer, h, ready):
        if not self._should_chunk_ffn(h):
            return super()._finish_ffn(model, layer, h, ready)
        out = torch.empty_like(h)
        for start in range(0, h.shape[1], self.ffn_rows):
            stop = min(start + self.ffn_rows, h.shape[1])
            residual = h[:, start:stop]
            out[:, start:stop].copy_(fused.add_scale(residual, model._mlp(layer, residual), layer.layer_scalar))
            ready()
        return out

    def _sliding(self, model, attn, x, cos, sin, ready):
        if hasattr(attn, "qkv_proj"):
            query, key, value = project_attention(attn, x, cos, sin)
        else:
            query = model._q(attn, x, cos, sin)
            key, value = model._kv(attn, x, cos, sin)
        window = model.sliding[0]
        segments = sliding_segments(x.shape[1], self.sliding_rows, window, x.device)
        out = torch.empty_like(x)
        ready()
        for start, stop, halo, q_cu, k_cu in segments:
            attended = sliding_attention.attend(
                query[0, start:stop], key[0, halo:stop], value[0, halo:stop],
                q_cu, k_cu, stop - start, stop - halo, attn.scaling, window,
            )
            out[:, start:stop].copy_(attn.o_proj(attended.reshape(1, stop - start, -1)))
            ready()
        return out

    def _preemptible_layer(self, model, index, h, cos, sin, p, ready):
        plan = p.plan
        split_sliding = model.types[index] == "sliding_attention" and 0 < self.sliding_rows < p.tokens
        if (not (split_sliding or self._should_chunk_ffn(h)) or len(plan.q_cu) != 2
                or plan.images is not None or model.causal_global or self._can_stripe_global(model, index, p)):
            return super()._preemptible_layer(model, index, h, cos, sin, p, ready)
        layer = model.layers[index]
        normalized = model.norm_input(layer.input_layernorm, h)
        if split_sliding:
            attn = self._sliding(model, layer.self_attn, normalized, cos, sin, ready)
        else:
            attn = model.attention(layer.self_attn, normalized, cos, sin, p)
        h = fused.residual_norm(h, layer.post_attention_layernorm, attn)
        ready()
        return self._finish_ffn(model, layer, h, ready)
