# SkillsBench through BenchFlow

KVBench keeps the model-serving boundary and delegates benchmark harness work
to the installed BenchFlow runtime.

## Runtime ownership

The request path is:

```text
BenchFlow agent
  -> HTTP request
  -> KVBench OpenAI-compatible endpoint
  -> Workload.next()
  -> Action(RUN, retainOutput=True)
  -> KVBench Engine / Worker
  -> Method.Run()
  -> Workload.observe()
  -> KVBench endpoint response
  -> BenchFlow agent
```

`helpers.endpoint.KVBenchEndpoint` is benchmark-independent. It owns the
OpenAI-compatible `/v1/chat/completions` server, chat-template rendering via
`ModelAdapter`, request queuing, tool-call parsing, response serialization,
and request/response logging. It does not know about task packages, skills,
agents, sandboxes, tools, or verifiers. `stream: true` is returned as a
complete OpenAI-compatible SSE response; generation itself still goes through
the normal KVBench Method path.

`helpers.benchflow.BenchflowRunner` is intentionally thin. It launches
`bench eval run`, passes through the selected source/agent/sandbox/skill mode,
waits for the process, and reads BenchFlow's official `result.json`. It does
not create a substitute verifier, reward file, trajectory, or result artifact.

## Configuration

The default configuration in `config.yaml` uses the pinned dataset source:

```yaml
AgentBenchFlow:
  SourceMode: dataset
  Dataset: skillsbench@1.1
  Agent: pi-acp
  Sandbox: docker
  SkillMode: with-skill
```

Use `SourceMode: local` with `SkillsBenchRepo` for a local checkout. The local
checkout is used for task enumeration and `--tasks-dir`; task parsing,
environment setup, skill provisioning, agent lifecycle, tool execution,
verification, and official artifacts remain BenchFlow responsibilities. The
`with-skill` and `no-skill` modes are passed to BenchFlow without changing the
agent prompt in KVBench.

For the currently installed BenchFlow 0.7.x, the equivalent command for one
dataset task is:

```bash
bench eval run \
  --dataset skillsbench@1.1 \
  --include citation-check \
  --agent pi-acp \
  --model vllm/Qwen3.8-27B \
  --sandbox docker \
  --skill-mode with-skill \
  --usage-tracking off \
  --jobs-dir /data/lyh/kvbench/outputs/benchflow/citation-check \
  --concurrency 1 \
  --agent-env BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:<kvbench-port>/v1 \
  --agent-env BENCHFLOW_PROVIDER_API_KEY=dummy
```

The model id defaults to the basename of KVBench's configured `ModelPath` and
is sent to BenchFlow as `vllm/<model-id>`. The Method controls generation
limits; the example `Main.py` configuration uses `maxNewTokens=4096` for the
agent workload.

BenchFlow 0.7.5 currently inserts its host-side LiteLLM provider proxy even
when usage tracking is disabled. In that setup the URL above is the proxy's
upstream URL: the proxy forwards to KVBench, and a Dockerized agent reaches
the proxy using this exact runtime-shaped URL:
`http://<docker-bridge-gateway>:<benchflow-host-proxy-port>/v1`.
Both values are dynamically selected by BenchFlow. If a future
BenchFlow release supports a direct agent-to-provider path, configure
`ProviderHost` to the Docker host gateway (for example
`host.docker.internal`) so the agent can reach the KVBench endpoint directly.
In both cases the inference request must terminate at KVBench's endpoint.

## Remote Docker runtime

Set ``Sandbox: remote-docker`` to keep KVBench and model execution on machine
A while moving the complete BenchFlow CLI + Docker runtime to machine B.  The
value is interpreted by KVBench and is never passed to BenchFlow; B still runs
the official command with ``--sandbox docker``.

```yaml
AgentBenchFlow:
  Sandbox: remote-docker
  RemoteDocker:
    Endpoint: http://127.0.0.1:8765
    # Required when B cannot reach an automatically detected address of A.
    # KVBenchAdvertiseHost: 10.0.0.21
    AuthTokenEnv: KVBENCH_REMOTE_TOKEN
    ConnectTimeoutSec: 10
    PollIntervalSec: 1
    ArtifactDownloadRetries: 3
```

For a same-machine A/B check, start the runtime in one terminal and KVBench in
another:

```bash
export KVBENCH_REMOTE_TOKEN='replace-with-a-random-shared-token'
python scripts/RemoteDockerRuntimeServer.py \
  --listen 127.0.0.1:8765 \
  --work-root /var/tmp/kvbench-remote-docker

# In the KVBench terminal, use the same token and Sandbox: remote-docker.
export KVBENCH_REMOTE_TOKEN='replace-with-a-random-shared-token'
python Main.py
```

The server is standalone and does not read ``config.yaml``. Its deployment
defaults (listen endpoint, work root, BenchFlow command, auth-token env name,
and concurrency limit) are at the top of
``scripts/RemoteDockerRuntimeServer.py`` and may also be overridden on the
command line.

At run creation A opens an authenticated KVBench provider endpoint and sends
its per-run URL to B. With BenchFlow 0.7.5 the traffic path is:

```text
B Docker agent -> B host LiteLLM proxy -> A KVBench endpoint -> KVBench Method
```

For two physical machines, A must bind its endpoint to a B-reachable interface
(the default is ``0.0.0.0``), and A's firewall must admit B. Set
``KVBenchAdvertiseHost`` explicitly when A is multi-homed or automatic route
detection chooses the wrong interface. The provider endpoint uses a random
per-run bearer token; the control API independently uses
``KVBENCH_REMOTE_TOKEN``. On an untrusted network, place this HTTP traffic on a
private network or encrypted tunnel.

``SourceMode: dataset`` needs no source upload. For ``SourceMode: local``, A
uploads only the selected ``tasks/<task-id>`` directory. B validates that the
task's declared Docker image is already present there; prepare SkillsBench
images on B before running. A does not need local Docker validation in remote
mode.

After BenchFlow exits, B packages its complete jobs directory. A downloads it,
checks its SHA-256, rejects unsafe archive entries, and extracts it below the
ordinary per-attempt output directory:

```text
<OutputDir>/<task-id>/run-.../
├── kvbench_llm_io.jsonl
└── remote-runtime/
    ├── benchflow.log
    └── .../result.json, agent/, verifier/, artifacts/, trajectory/
```

The existing recursive official-result lookup and scoring path are unchanged.
This first implementation deliberately omits leases/heartbeats, live log
tailing, resumable downloads, and scheduling across multiple B hosts.

## Results and manual checks

Each case writes BenchFlow's official artifacts below `OutputDir`. KVBench
places the complete official result payload in `Result.output` and copies
selected fields into `Result.metadata`; scoring reads canonical reward/score
fields from that payload. A failed BenchFlow rollout is isolated to its case:
that case receives a zero score, its error remains in diagnostics, and the
next case continues on the same method worker.

Suggested staged checks:

1. Check whether the installed runtime has the optional ground-truth agent:

   ```bash
   bench agent list | rg '(^|[[:space:]])oracle([[:space:]]|$)'
   ```

   If it is listed, validate the task package and verifier without KVBench:

   ```bash
   bench eval run --dataset skillsbench@1.1 --include citation-check \
     --agent oracle --sandbox docker --skill-mode with-skill \
     --usage-tracking off --jobs-dir /tmp/benchflow-ground-truth
   ```

2. Run the endpoint/runner boundary tests:

   ```bash
   uvx --with pytest --with pyyaml --with rich --with transformers --with jinja2 \
     pytest -q tests/test_agent_benchflow.py
   ```

3. Run a real single-task smoke test after Docker, the model, and the selected
   BenchFlow agent are available:

   ```bash
   python Main.py
   ```

4. Inspect both the official BenchFlow `result.json` and the KVBench JSONL
   request/response log under the configured output directory. The latter is
   the place to extend exact TTFT, token throughput, KV reuse, retained-output
   reuse, batching, or multi-agent scheduling measurements.
