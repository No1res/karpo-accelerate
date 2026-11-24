# Karpo Accelerate

Hardware-aware serving for [Karpo Decision 31B](https://huggingface.co/MachinePulseAI/karpo-decision-31B). One request contains one new context and one choice question. The service runs the decision forward and returns the selected option.

The server batches independent requests with separate attention boundaries, replays token-bucket CUDA graphs, and lets short batches run between blocks of long contexts. On A100 it can use W8A8 projection kernels or a mixed-precision profile with rotated W4A4 feed-forward layers and W8A8 attention projections. Per-task linear heads let confident choices exit at an intermediate layer.

## Install

Use Linux with Python 3.12 and a CUDA GPU. The measured quantized configuration used A100-SXM4-80GB, PyTorch 2.14.0+cu126, Transformers 5.17.0 and CUDA Toolkit 12.8 for compilation. The initial model load uses BF16 weights before replacing projections, so plan GPU memory for startup as well as steady state.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install '.[cuda]'
```

The CUDA extra pins the Karpo inference API to [revision 6600011e](https://github.com/machinepulse-ai/karpo-decision/tree/6600011eb5cf46232d056ec0717dc726146d29de). Installing `pip install .` gives you the benchmark and replica gateway on a machine without a GPU. See the [PyTorch wheel index instructions](https://pytorch.org/get-started/previous-versions/) for compatible CUDA builds.

Download the measured model revision to a local cache and set its path. The model is about 63 GB on disk.

```bash
karpo_checkpoint=$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    "MachinePulseAI/karpo-decision-31B",
    revision="b8e6fd7463cea8662c556d6257955d7ef9c10d19",
))
PY
)
```

## Run a replica

Start with BF16 dynamic batching. This profile uses the original projections and needs only the installed CUDA runtime.

```bash
CUDA_VISIBLE_DEVICES=0 karpo-accelerate serve --checkpoint "$karpo_checkpoint" --port 18440
```

From another terminal, send a [sample request](examples/request.json).

```bash
curl -fsS -H 'Content-Type: application/json' \
  --data-binary @examples/request.json http://127.0.0.1:18440/v1/systemone
```

Example response

```json
{"answers":{"decision":{"choice":"billing"}}}
```

`GET /healthz` shows the loaded profile and graph settings. Pass `--full-response` for the original option scores on complete forwards.

### A100 mixed precision

Install a CUDA toolkit with `nvcc`, plus a C++ compiler. Set `CUDA_HOME` if it is outside the usual search path. The build command fetches and checksum-verifies the pinned CUTLASS 4.2.1 and vLLM 0.19.1 headers. For an offline build, pass `--cutlass-include` and `--vllm-source` with local copies.

```bash
karpo-accelerate build-kernels --profile w4a4
CUDA_VISIBLE_DEVICES=0 karpo-accelerate serve \
  --checkpoint "$karpo_checkpoint" --profile w4a4 --port 18440
```

The quantized profiles enable 4,096-token graphs and send sequences longer than 1,536 tokens to the cooperative long-context path. Use `--graph-tokens 0` or `--long-tokens 0` to isolate the corresponding ablation. All profiles default to eight requests or 4,096 tokens per ordinary batch and a 3 ms collection window. Oversize sequences follow the full-context path.

Pass a [compact-rubrics file](examples/compact-rubrics.json) with `--compact-rubrics` to replace the duplicated front rubric with a short preview or remove it with `null`. A rule applies to the exact instruction string in a plain-text, single-choice request. The full rubric beside the answer choices stays in the input.

### Calibrated early exits

Run `karpo-accelerate calibrate` on your own disjoint labelled data to fit one or two intermediate-layer heads with the backbone frozen. Use the resulting artifact with `serve --exit-heads` in choice-only mode. See [the data format and calibration workflow](docs/early-exit.md).

### Multiple GPUs

Start one replica per GPU, assigning each a distinct port. Repeat the same profile, rubric and head parameters on each replica.

```bash
CUDA_VISIBLE_DEVICES=0 karpo-accelerate serve --checkpoint "$karpo_checkpoint" --profile w4a4 --port 18441
# In a separate shell with karpo_checkpoint set
CUDA_VISIBLE_DEVICES=1 karpo-accelerate serve --checkpoint "$karpo_checkpoint" --profile w4a4 --port 18442
# In a third shell
karpo-accelerate proxy --port 18440 \
  --replica http://127.0.0.1:18441 --replica http://127.0.0.1:18442
```

The gateway sends each request to the replica with the fewest outstanding requests. The HTTP processes bind to loopback by default. Put authentication and TLS in your own gateway before exposing a service on a shared network.

## Measure

```bash
karpo-accelerate bench --replica http://127.0.0.1:18440 \
  --requests examples/requests.jsonl --qps 2 --seconds 2 --arrivals uniform
```

The benchmark reports HTTP P50/P95, scheduled-arrival P95, achieved QPS, failures, and quality metrics when labels are present. The [benchmark guide](docs/benchmarks.md) describes the input format, an original-service comparison, and the historical four-A100 results.

Run `python -m unittest discover -s tests -v` for the CPU checks. After building the kernels, set `KARPO_KERNEL_DIR=build/kernels` on an A100 and rerun the same command to check INT8 and INT4 operator outputs.

## License

The code in this repository is Apache-2.0. Model weights come from the [model repository](https://huggingface.co/MachinePulseAI/karpo-decision-31B) under its own terms. See [NOTICE](NOTICE) for upstream credits.
