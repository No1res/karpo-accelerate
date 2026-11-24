"""Open-loop latency and quality measurement on fresh, independent requests."""

import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from time import perf_counter, sleep

from .cases import read_cases
from .routing import Replicas


def arrivals(mode, qps, duration, seed):
    count = round(qps * duration)
    if count < 1:
        raise ValueError("The requested window must contain at least one decision")
    if mode == "uniform":
        return [i * duration / count for i in range(count)]
    rng = random.Random(seed)
    return sorted(rng.random() * duration for _ in range(count))


def workload(cases, count, seed, repeat):
    pool = list(cases)
    random.Random(seed).shuffle(pool)
    if repeat:
        return [pool[i % len(pool)] for i in range(count)]
    unique, seen = [], set()
    for item in pool:
        state = item.request["state"]
        if state not in seen:
            unique.append(item)
            seen.add(state)
    if len(unique) < count:
        raise ValueError(f"Need {count} distinct contexts, found {len(unique)}. Lower the load or supply more cases.")
    return unique[:count]


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    rank = p * (len(ordered) - 1)
    left, right = math.floor(rank), math.ceil(rank)
    return ordered[left] + (rank - left) * (ordered[right] - ordered[left])


def send(replicas, case, body, scheduled):
    prediction, error = None, None
    with replicas.reserve() as slot:
        started = perf_counter()
        try:
            status, payload, _ = replicas.request(slot, "POST", "/v1/systemone", body,
                                                  {"Content-Type": "application/json"})
            if status == 200:
                prediction = json.loads(payload)["answers"][case.question_id]["choice"]
                if not isinstance(prediction, str) or prediction not in case.question["criteria"]:
                    raise ValueError("Invalid choice")
            else:
                error = f"HTTP {status}"
        except (HTTPException, OSError) as exc:
            error = type(exc).__name__
        except (ValueError, KeyError, TypeError):
            error = "invalid_response"
        done = perf_counter()
    return {"id": case.id, "choice": prediction if error is None else None, "expected": case.expected,
            "error": error, "replica": slot, "completed_at": done,
            "http_ms": 1000 * (done - started), "arrival_ms": 1000 * (done - scheduled),
            "send_delay_ms": 1000 * (started - scheduled)}


def summarize(rows, start, duration, deadline):
    success = [row for row in rows if row["error"] is None]
    labelled = [row for row in rows if row["expected"] is not None]
    correct = [row for row in labelled if row["choice"] == row["expected"]]
    last = max(row["completed_at"] for row in rows)
    elapsed = max(duration, last - start)
    matrix = Counter((row["expected"], row["choice"]) for row in labelled)
    return {
        "requests": len(rows),
        "failed": len(rows) - len(success),
        "completed_qps": len(success) / elapsed,
        "window_completed_qps": sum(row["completed_at"] <= start + duration for row in success) / duration,
        "drain_seconds": max(0, elapsed - duration),
        "http_p50_ms": percentile([row["http_ms"] for row in success], 0.50),
        "http_p95_ms": percentile([row["http_ms"] for row in success], 0.95),
        "arrival_to_response_p95_ms": percentile([row["arrival_ms"] for row in success], 0.95),
        "send_delay_p95_ms": percentile([row["send_delay_ms"] for row in rows], 0.95),
        "labelled": len(labelled),
        "accuracy": len(correct) / len(labelled) if labelled else None,
        "deadline_ms": deadline,
        "timely_correct_rate": sum(row["http_ms"] <= deadline for row in correct) / len(labelled) if labelled else None,
        "confusion": [{"expected": expected, "choice": choice, "count": count}
                      for (expected, choice), count in sorted(matrix.items(), key=lambda entry: str(entry[0]))],
    }


def run_benchmark(args):
    offsets = arrivals(args.arrivals, args.qps, args.seconds, args.seed)
    cases = workload(read_cases(args.requests), len(offsets), args.seed, args.repeat)
    payloads = [json.dumps(case.request, ensure_ascii=False).encode() for case in cases]
    replicas = Replicas(args.replica, args.timeout)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            start = perf_counter()
            futures = []
            for offset, case, payload in zip(offsets, cases, payloads, strict=True):
                scheduled = start + offset
                wait = scheduled - perf_counter()
                if wait > 0:
                    sleep(wait)
                futures.append(workers.submit(send, replicas, case, payload, scheduled))
            rows = [future.result() for future in futures]
    finally:
        replicas.close()
    report = {"offered_qps": args.qps, "duration_seconds": args.seconds, "arrivals": args.arrivals, "seed": args.seed,
              **summarize(rows, start, args.seconds, args.deadline_ms)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
