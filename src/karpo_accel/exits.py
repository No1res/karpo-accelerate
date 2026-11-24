"""Run task-specific decision heads and replay the deeper graphs for pending requests."""

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F
from karpo_decision.answers import answer
from karpo_decision.model.packed import pack
from karpo_decision.serve import server

from .cases import schema_key
from .graphs import CascadeGraphs
from .long_context import ChunkedRunner


def marker_features(model, hidden, markers):
    return F.normalize(model.lm.norm(hidden.index_select(1, markers))[0].float())


@dataclass(frozen=True, slots=True)
class Head:
    slot: int
    choices: tuple[str, ...]
    cutoffs: tuple[float | None, ...]


class ExitPolicy:
    def __init__(self, records, hidden_size, device):
        self.heads = {}
        max_choices = max(len(record["criteria"]) for record in records)
        weights, biases = [], []
        for slot, record in enumerate(records):
            criteria = tuple(tuple(pair) for pair in record["criteria"])
            choices = tuple(name for name, _ in criteria)
            signature = record["instructions"], criteria
            cutoffs = tuple(record["cutoffs"])
            weight, bias = record["weight"].float(), record["bias"].float()
            n = len(choices)
            if len(set(choices)) != n or not 2 <= n <= 255 or signature in self.heads:
                raise ValueError("An exit artifact has duplicate rubrics or invalid options")
            if weight.shape != (hidden_size, n) or bias.shape != (n,):
                raise ValueError("An exit head has a different shape from the checkpoint")
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError("An exit head contains nonfinite coefficients")
            if len(cutoffs) != n or any(x is not None and not (0.0 <= x < float("inf")) for x in cutoffs):
                raise ValueError("An exit head has invalid confidence cutoffs")
            self.heads[signature] = Head(slot, choices, cutoffs)
            padding = max_choices - n
            weights.append(F.pad(weight.T.contiguous(), (0, 0, 0, padding)))
            biases.append(F.pad(bias, (0, padding), value=float("-inf")))
        self.weights = torch.stack(weights).to(device)
        self.biases = torch.stack(biases).to(device)

    def match(self, job):
        key, = job.plan[0].keys
        question = job.request["questions"][key]
        if question["type"] == "choice":
            return self.heads.get(schema_key(question))
        return None

    def classify(self, model, hidden, packed, jobs):
        candidates = [(idx, head) for idx, job in enumerate(jobs) if (head := self.match(job)) is not None]
        if not candidates:
            return {}
        rows, heads = zip(*candidates, strict=True)
        row_indices = torch.tensor(rows, device=hidden.device)
        slots = torch.tensor([head.slot for head in heads], device=hidden.device)
        features = marker_features(model, hidden, packed.markers.index_select(0, row_indices))
        logits = torch.bmm(features[:, None], self.weights.index_select(0, slots))[:, 0]
        values, indices = (logits + self.biases.index_select(0, slots)).topk(2, dim=1)
        scores = torch.stack((indices[:, 0].float(), values[:, 0] - values[:, 1]), dim=1).cpu().tolist()
        choices = {}
        for row, head, (index, confidence) in zip(rows, heads, scores, strict=True):
            cutoff = head.cutoffs[int(index)]
            if cutoff is not None and confidence > cutoff:
                choices[row] = head.choices[int(index)]
        return choices


class ExitRunner(ChunkedRunner):
    def __init__(self, decider, device, *, graphs, policies, **options):
        self.graphs = graphs
        self.policies = policies
        super().__init__(decider, device, **options)

    @staticmethod
    def complete(job, result):
        key, = job.plan[0].keys
        job.result.set_result({"answers": {key: result},
                               "usage": {"input_tokens": job.tokens, "output_tokens": 0, "sequences": 1}})

    def _exit(self, policy, packed, hidden, jobs):
        early = policy.classify(self.decider.model, hidden, packed, jobs)
        for i, choice in early.items():
            self.complete(jobs[i], {"type": "choice", "choice": choice})
        return [i for i in range(len(jobs)) if i not in early]

    def _continue(self, packed, key, hidden, jobs, kept, forward):
        if len(kept) == len(jobs):
            return jobs, packed, key, forward(packed, key, hidden)
        pending = [jobs[i] for i in kept]
        compact = pack([job.plan[0] for job in pending], self.decider.ckpt.backbone, self.device)
        _, compact_key = self.graphs.bucket(compact)
        # The original bucket also fits the subset if padding exceeds the normal limit.
        compact_key = compact_key or key
        result = forward(compact, compact_key, hidden, packed.cu_cpu.tolist(), kept)
        return pending, compact, compact_key, result

    @torch.inference_mode()
    def _run_batch(self, jobs):
        if any(not job.packable or len(job.plan[0].keys) != 1 for job in jobs):
            return super()._run_batch(jobs)
        if not any(policy.match(job) for policy in self.policies for job in jobs):
            return super()._run_batch(jobs)
        decider, graphs = self.decider, self.graphs
        started = perf_counter()
        try:
            packed = pack([job.plan[0] for job in jobs], decider.ckpt.backbone, self.device)
            _, key = graphs.bucket(packed)
            if key is None:
                return super()._run_batch(jobs)
            active = jobs
            if isinstance(graphs, CascadeGraphs):
                hidden = graphs.start_first(packed, key)
                kept = self._exit(self.policies[0], packed, hidden, active)
                if not kept:
                    server.log(event="packed_early_exit", requests=len(jobs), full_depth=0)
                    self._log_batch(jobs, started)
                    return
                active, packed, key, hidden = self._continue(packed, key, hidden, active, kept, graphs.advance)
            else:
                hidden = graphs.start(packed, key)

            kept = self._exit(self.policies[-1], packed, hidden, active)
            if kept:
                pending, _, _, final = self._continue(packed, key, hidden, active, kept, graphs.finish)
                logits = decider.readout(final).cpu()
                for job, row in zip(pending, logits, strict=True):
                    question = job.request["questions"][job.plan[0].keys[0]]
                    scores = row[:job.plan[0].counts[0]] / decider.config.temperatures[question["type"]]
                    self.complete(job, answer(question, scores.softmax(-1).tolist()))
            server.log(event="packed_early_exit", requests=len(jobs), full_depth=len(kept))
            self._log_batch(jobs, started)
        except Exception as error:
            server.log(event="early_exit_fallback", error=type(error).__name__)
            pending = [job for job in jobs if not job.result.done()]
            if pending:
                super()._run_batch(pending)
