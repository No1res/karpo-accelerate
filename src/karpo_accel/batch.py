"""Batch independent Karpo requests without sharing their attention spans."""

from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Any

import torch
from karpo_decision.encoding import Encoded
from karpo_decision.serve import server


@dataclass(slots=True)
class Job:
    request: dict[str, Any]
    plan: list[Encoded]
    tokens: int
    received: float
    result: Future[dict[str, Any]] = field(default_factory=Future)

    @property
    def packable(self) -> bool:
        return len(self.plan) == 1 and not self.plan[0].images


class BatchRunner(server.Runner):
    def __init__(self, decider, device, *, max_requests: int, max_tokens: int, wait_ms: float):
        super().__init__(decider, device)
        self.max_requests = max_requests
        self.max_tokens = min(max_tokens, decider.token_budget)
        self.wait_seconds = wait_ms / 1000
        self.pending: Queue[Job] = Queue(maxsize=server.Server.request_queue_size)
        Thread(target=self._work, daemon=True, name="karpo-batch-worker").start()

    def run(self, questions: dict[str, Any], plan: list[Encoded]) -> dict[str, Any]:
        job = Job({"state": None, "questions": questions}, plan, sum(part.length for part in plan), perf_counter())
        self.pending.put(job)
        return job.result.result()

    def _work(self) -> None:
        torch.cuda.set_device(self.device)
        next_job = None
        while True:
            first = next_job if next_job is not None else self.pending.get()
            jobs, next_job = self._collect(first)
            self._run_batch(jobs)

    def _collect(self, first: Job) -> tuple[list[Job], Job | None]:
        jobs = [first]
        total = first.tokens
        if not first.packable or total >= self.max_tokens:
            return jobs, None

        deadline = first.received + self.wait_seconds
        while len(jobs) < self.max_requests and total < self.max_tokens:
            try:
                job = self.pending.get(timeout=max(0.0, deadline - perf_counter()))
            except Empty:
                break
            if not job.packable or total + job.tokens > self.max_tokens:
                return jobs, job
            jobs.append(job)
            total += job.tokens
        return jobs, None

    def _run_batch(self, jobs: list[Job]) -> None:
        if len(jobs) > 1:
            started = perf_counter()
            try:
                # Karpo retains each plan's sequence boundaries and reads out its own markers.
                responses = self.decider.decide([job.request for job in jobs], plans=[job.plan for job in jobs])
                if len(responses) != len(jobs):
                    raise ValueError("Karpo returned the wrong number of responses")
            except Exception as error:
                server.log(event="batch_fallback", size=len(jobs), error=type(error).__name__)
            else:
                self._log_batch(jobs, started)
                for job, response in zip(jobs, responses, strict=True):
                    job.result.set_result(response)
                return

        for job in jobs:
            started = perf_counter()
            try:
                response = super().run(job.request["questions"], job.plan)
            except Exception as error:
                job.result.set_exception(error)
            else:
                self._log_batch([job], started)
                job.result.set_result(response)

    @staticmethod
    def _log_batch(jobs: list[Job], started: float) -> None:
        server.log(event="batch", size=len(jobs), tokens=sum(job.tokens for job in jobs),
                   run_ms=round((perf_counter() - started) * 1000, 2))


class TopChoiceHandler(server.Handler):
    def _send(self, status, body, timing=None):
        answers = body.get("answers", {})
        if status == 200 and len(answers) == 1:
            key, answer = next(iter(answers.items()))
            if answer.get("type") == "choice":
                body = {"answers": {key: {"choice": answer["choice"]}}}
        super()._send(status, body, timing)
