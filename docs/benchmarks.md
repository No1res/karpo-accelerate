# Benchmark Karpo serving

## Request file

Supply UTF-8 JSONL. Each row holds a full Karpo HTTP request under `request`, with a text state and exactly one choice question. Use `id` to identify your cases and `expected` for the correct option. [Example request rows](../examples/requests.jsonl).

```json
{"id":"case-1","request":{"state":"My invoice has a duplicate charge.","questions":{"decision":{"type":"choice","instructions":"Route the message to the right team.","criteria":{"billing":"Invoices and payments","support":"Technical problems"}}}},"expected":"billing"}
```

The benchmark selects distinct states within each window. Set `--repeat` only when repeated contexts are intentional, such as a short smoke test. A 180-second window at 10 QPS needs 1,800 distinct states. Identical request files and seeds give the same ordered inputs and planned arrival times. The Poisson option samples arrivals conditional on the number of requests in the window.

## Compare configurations

Start the [upstream service](https://github.com/machinepulse-ai/karpo-decision/tree/6600011eb5cf46232d056ec0717dc726146d29de) and the optimized service in separate runs on the same available GPU. Use the same checkpoint, arrival mode, duration, request file and seed.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 \
  -m karpo_decision.serve.server --checkpoint "$karpo_checkpoint" --port 18440
```

```bash
karpo-accelerate bench --replica http://127.0.0.1:18440 \
  --requests workload.jsonl --qps 10 --seconds 180 --arrivals poisson --seed 20261121
```

HTTP latency starts when the client sends a POST and ends when it parses the choice. It includes server-side queueing. Scheduled-arrival latency includes delay before a client worker can send. Under saturation, inspect HTTP P95 alongside window-completed QPS, drain time, arrival-to-response P95 and failures. With labels, the deadline rate is the share of all labelled requests that are both correct and returned on time. The confusion matrix supports per-option false-positive and false-negative counts.

To compare replica counts without an HTTP gateway hop, pass one `--replica` per server to the benchmark. Give all replicas the same serving options.

## Historical A100 results

The original experiment used the Karpo 31B revision from the README, a mixture of Jev600 and safety decisions, and one new text context per request. The final path combined compact task rubrics, rotated W4A4 FFNs, W8A8 attention projections, 4,096-token CUDA graphs, cooperative long-context scheduling and a task-specific layer 34/42 cascade.

At offered 10 QPS on one A100, the stock service completed 5.38–5.61 QPS. Its HTTP P95 was 24.47–25.25 s. The optimized service completed 9.99–10.00 QPS with an HTTP P95 of 120.8–234.0 ms. The stock measurements included queue buildup.

| Four independent A100 replicas | Offered QPS | Each run | HTTP P95 on two seeds |
|---|---:|---:|---:|
| Poisson | 72 | 165 s | 244.9 / 235.4 ms |
| Uniform | 76 | 155 s | 222.2 / 219.1 ms |

Both rates passed with zero failed requests, at least 98% of offered requests completed within the window, and ≤1 s drain time. The next 4 QPS grid points exceeded 300 ms in 75-second screens. Full-set replay for that profile scored 592/600 on Jev600 and 86.13% on the safety tasks. The historical heads used task-specific fitting and calibration. To assess accuracy and latency on your data, fit your own heads and run the paired comparison above.
