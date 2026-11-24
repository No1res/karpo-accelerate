# Train and calibrate an early exit

The model's original decision marker gives one feature vector per question. The calibration command captures it at the requested body layers, normalizes it, and fits a ridge-regularized linear classifier for each rubric. Serving computes the margin between the best two options. When the margin clears the option's threshold, that request can finish and the later CUDA graph stages process the remaining requests.

Use three JSONL files with the [same row format as the benchmark](benchmarks.md#request-file). Set `expected` to the correct option key in every row. Every split must represent the same rubrics and ordered criteria. The fit split needs an example of every option. Use distinct state text across the three splits, and keep your final deployment evaluation independent of all three. With default W4A4 settings, use contexts that encode to at most 1,536 tokens for calibration.

```bash
karpo-accelerate calibrate --checkpoint "$karpo_checkpoint" --profile w4a4 \
  --depths 34 42 --fit fit.jsonl --calibration calibration.jsonl \
  --holdout holdout.jsonl --output heads.pt

CUDA_VISIBLE_DEVICES=0 karpo-accelerate serve \
  --checkpoint "$karpo_checkpoint" --profile w4a4 --exit-heads heads.pt
```

Both commands should use the same checkpoint revision, precision, compact-rubric file and batch limits. Use one layer with `--depths 42`. Depth means the number of completed body layers, starting at one. Stages are fit and calibrated from deeper to earlier. Each stage reserves uncertain cases for the decisions that would have followed it.

By default, each option's acceptance threshold is the largest margin at which that option disagreed with the existing downstream choice on the calibration split, plus `--margin`. An option with no calibration predictions remains on the downstream path. Add `--priority-option harmful` to calibrate that option against labelled false positives instead. Review accuracy, false positives and false negatives in the printed holdout confusion matrices before choosing a policy. A threshold calibrated on a finite set needs a separate deployment-domain check.

The artifact stores head tensors, per-option thresholds, runtime compatibility metadata and aggregate holdout results. The server compares model metadata, the shard layout, precision, rubric settings and batch limits before loading a head. A task matches on the complete instruction string and the exact order and text of its criteria. Requests outside an exit's rubric or graph eligibility can use the complete forward.
