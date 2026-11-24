"""Fit linear heads on a frozen Karpo model and calibrate per-option exits."""

import json
import os
from collections import Counter, defaultdict
from tempfile import NamedTemporaryFile

import torch

from .cases import read_cases, schema_key


def fit_linear(features, labels, classes, ridge=0.05):
    mean = features.mean(0)
    centered = features - mean
    targets = torch.eye(classes, dtype=features.dtype, device=features.device)[labels]
    prior = targets.mean(0)
    if len(features) <= features.shape[1]:
        gram = centered @ centered.T
        gram.diagonal().add_(ridge)
        weights = centered.T @ torch.linalg.solve(gram, targets - prior)
    else:
        gram = centered.T @ centered
        gram.diagonal().add_(ridge)
        weights = torch.linalg.solve(gram, centered.T @ (targets - prior))
    return weights, prior - mean @ weights


def thresholds(predictions, fallback, expected, choices, priority, margin):
    cutoffs = []
    for choice in choices:
        observed = False
        max_error = 0.0
        use_labels = choice in priority and any(label != choice for label in expected)
        reference = expected if use_labels else fallback
        for (prediction, confidence), target in zip(predictions, reference, strict=True):
            if prediction == choice:
                observed = True
                if prediction != target:
                    max_error = max(max_error, confidence)
        cutoffs.append(max_error + margin if observed else None)
    return cutoffs


def read_splits(args):
    states = {}
    splits = {}
    keys = None
    for name in ("fit", "calibration", "holdout"):
        cases = read_cases(getattr(args, name), labels=True)
        for case in cases:
            state = case.request["state"]
            if states.setdefault(state, name) != name:
                raise ValueError("Fit, calibration, and holdout must not share the same state text")
        tasks = {schema_key(case.question) for case in cases}
        if keys is not None and tasks != keys:
            raise ValueError("Every rubric must occur with the same ordered criteria in all three splits")
        keys = tasks
        splits[name] = cases
    return splits


def batches(decider, cases, args):
    jobs, count = [], 0
    max_batch_tokens = min(args.batch_tokens, decider.token_budget)
    max_sequence = args.long_tokens or args.max_length
    for case in cases:
        plan = decider.plan(case.request)
        if len(plan) != 1 or plan[0].images or len(plan[0].keys) != 1:
            raise ValueError(f"{case.id} did not encode as one text choice")
        sequence = plan[0]
        if sequence.length > max_sequence:
            raise ValueError(f"{case.id} exceeds the early-exit length limit of {max_sequence} tokens")
        if jobs and (count + sequence.length > max_batch_tokens or len(jobs) == args.batch_requests):
            yield jobs
            jobs, count = [], 0
        jobs.append((case, sequence))
        count += sequence.length
    if jobs:
        yield jobs


@torch.inference_mode()
def capture(decider, cases, args):
    from karpo_decision.answers import answer
    from karpo_decision.families.gemma.packed import block_plan
    from karpo_decision.model.packed import pack

    from .exits import marker_features

    model = decider.model
    records = defaultdict(list)
    for jobs in batches(decider, cases, args):
        packed = pack([sequence for _, sequence in jobs], decider.ckpt.backbone, decider.device)
        packed.plan = block_plan(packed, model.window)
        hidden = model.embed(packed.input_ids)
        rope = model.rope(hidden, packed.seqpos)
        features = {}
        for i, layer_type in enumerate(model.types[:-1]):
            hidden = model.run(i, hidden, *rope[layer_type], packed)
            if i + 1 in args.depths:
                features[i + 1] = marker_features(model, hidden, packed.markers).cpu()
        logits = decider.readout(model.head(hidden, rope, packed)).cpu()
        for row, (case, sequence) in enumerate(jobs):
            scores = logits[row, :sequence.counts[0]] / decider.config.temperatures["choice"]
            full = answer(case.question, scores.softmax(-1).tolist())["choice"]
            records[schema_key(case.question)].append({
                "expected": case.expected, "full": full, "choice": full, "exit_depth": None,
                "features": {depth: value[row] for depth, value in features.items()},
            })
    return records


def predict(rows, depth, weights, bias, choices):
    features = torch.stack([row["features"][depth] for row in rows])
    scores, indices = (features @ weights + bias).topk(2, dim=1)
    return [(choices[index], confidence) for index, confidence
            in zip(indices[:, 0].tolist(), (scores[:, 0] - scores[:, 1]).tolist(), strict=True)]


def train(records, depths, priority, margin):
    stages = []
    for depth in reversed(depths):
        heads = []
        for instructions, criteria in sorted(records["fit"]):
            task = instructions, criteria
            fit, calibration, holdout = (records[name][task] for name in ("fit", "calibration", "holdout"))
            choices = tuple(key for key, _ in criteria)
            if {row["expected"] for row in fit} != set(choices):
                raise ValueError(f"The fitting split for {instructions[:80]!r} needs a label for each option")
            features = torch.stack([row["features"][depth] for row in fit])
            labels = [choices.index(row["expected"]) for row in fit]
            weight, bias = fit_linear(features, labels, len(choices))
            prediction = predict(calibration, depth, weight, bias, choices)
            cutoff = thresholds(prediction, [row["choice"] for row in calibration],
                                [row["expected"] for row in calibration], choices, priority, margin)
            if all(value is None for value in cutoff):
                continue
            for rows in (calibration, holdout):
                for row, (choice, confidence) in zip(rows, predict(rows, depth, weight, bias, choices), strict=True):
                    limit = cutoff[choices.index(choice)]
                    if limit is not None and confidence > limit:
                        row["choice"], row["exit_depth"] = choice, depth
            heads.append({"instructions": instructions, "criteria": list(criteria), "weight": weight.contiguous(),
                          "bias": bias.contiguous(), "cutoffs": cutoff})
        if heads:
            stages.append({"depth": depth, "heads": heads})
    return list(reversed(stages))


def holdout_summary(groups):
    report = []
    for (instructions, criteria), rows in sorted(groups.items()):
        outcomes = {}
        for name in ("full", "choice"):
            matrix = Counter((row["expected"], row[name]) for row in rows)
            outcomes[name] = [{"expected": expected, "choice": choice, "count": count}
                              for (expected, choice), count in sorted(matrix.items())]
        report.append({
            "instructions": instructions, "options": [key for key, _ in criteria], "requests": len(rows),
            "full_correct": sum(row["expected"] == row["full"] for row in rows),
            "correct": sum(row["expected"] == row["choice"] for row in rows),
            "changed": sum(row["full"] != row["choice"] for row in rows),
            "early_exits_by_depth": dict(Counter(row["exit_depth"] for row in rows if row["exit_depth"] is not None)),
            "confusion": outcomes,
        })
    return report


def calibrate(args):
    from .runtime import identity, load_model

    splits = read_splits(args)
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists")
    decider, rubrics = load_model(args)
    if not getattr(decider.model, "prune_last", False) or args.depths[-1] >= len(decider.model.layers) - 1:
        raise ValueError("Calibration depths must be inside Karpo's marker-pruned Gemma body")
    records = {}
    for name, cases in splits.items():
        records[name] = capture(decider, cases, args)
        print(f"Captured {len(cases)} cases for {name}", flush=True)
    stages = train(records, args.depths, set(args.priority_option), args.margin)
    if not stages:
        raise ValueError("No eligible heads were fit")
    summary = holdout_summary(records["holdout"])
    artifact = {"format": 1, "runtime": identity(args, rubrics), "stages": stages, "holdout": summary}
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
        path = stream.name
        try:
            torch.save(artifact, stream)
            stream.flush()
            os.link(path, args.output)
        finally:
            os.unlink(path)
    print(json.dumps({"artifact": str(args.output), "holdout": summary}, ensure_ascii=False, indent=2))
