# Task Interface and Extension

A Task defines what to evaluate: it produces samples, builds their execution workflows, and scores the output according to task-specific quality criteria. System metrics such as TTFT and throughput are handled by `metrics/`.

## Interface and data structures

Every task inherits from `core.Task.Task` and implements:

```python
def Cases(self) -> Iterator[Case]: ...
def Evaluate(self, result: Result, metadata: dict) -> dict[str, float]: ...
```

The `Task` constructor accepts optional `tag` and `maxNewTokens` arguments. A task can also override `defaultMaxNewTokens`. On each RUN action, the Engine passes the task's `maxNewTokens` value to the Method.

Each `Case` has three parts:

- `input`: task payload, with a type defined by the selected Workflow.
- `workflow`: stateful execution policy that turns the input into Method actions.
- `metadata`: information needed for scoring, such as reference answers, sample IDs, or other evaluation data.

`Evaluate` receives the final RUN `Result` and its `metadata`, then returns a mapping of metric names to numeric scores, for example `{"accuracy": 1.0}`. The Engine aggregates each metric across Cases for that Task. Even when a task has only one score, return a one-key mapping so the report retains a meaningful metric name.

## Current tasks

| Class | Dataset/scenario | Main quality metrics |
| --- | --- | --- |
| `NIAHTask` / `NIAHShuffleTask` | RULER needle-in-a-haystack; the shuffle variant permutes reusable segments | Accuracy / string match |
| `CWETask` / `CWEShuffleTask` | RULER common-words extraction | Accuracy |
| `VTTask` / `VTShuffleTask` | RULER variable tracking | Accuracy |
| `MusiqueTask` | MuSiQue multi-hop QA | Token F1 and exact match (the paper reports F1) |
| `TwoWikiMultiHopQATask` | 2WikiMultiHopQA multi-hop QA | Token F1 and exact match (the paper reports F1) |
| `HotpotQATask` | HotpotQA multi-hop QA | Token F1 and exact match (the paper reports F1) |
| `TriviaQATask` | TriviaQA QA | Token F1 and exact match (the paper reports F1) |
| `SamsumTask` | Dialogue summarization | ROUGE-L |
| `MultiNewsTask` | Multi-document summarization | ROUGE-L |
| `GovReportTask` | Long-document summarization | ROUGE-L |
| `KVCommMMLUTask` / `KVCommGSM8KTask` / `KVCommHumanEvalTask` / `KVCommCopyTask` | Multi-agent communication and synthetic copy task | Accuracy or task score |
| `FreshGapTask` | Synthetic check with fresh text inserted between two reusable segments | Accuracy |
| `AgentBenchFlowTask` | Agent execution and skill reuse through SkillsBench/BenchFlow | BenchFlow reward/score |

Task classes are exported from `tasks/__init__.py`. Most public dataset locations are controlled by `DatasetPath`; see `tasks/bases/RulerBase.py` and `tasks/bases/KBBase.py` for the RULER, knowledge-base, and LongBench-style schemas.

## Using an existing task

Instantiate tasks in `Main.py` and pass them to the same Engine as the methods. Dataset size is usually controlled by `maxSamples` or task-specific arguments. For example:

```python
GovReportTask(maxSamples=64, nChunks=4, maxSampleLength=32768)
```

Data locations, model paths, and dataset preparation are configured in the root `config.yaml` and `scripts/PrepareDataset.py`.

Most RAG tasks use `workflow.RAGWorkflow.RAGInput`: `prepare_input` contains reusable segments and `run_input` is the complete target prompt. With no warm-up, set `prepare_input=[]`; the Workflow will issue only a RUN action.

## Adding a task

1. Create a module under `tasks/` and subclass `Task` (or a suitable shared base class). Set `name` and an appropriate `defaultMaxNewTokens`.
2. Validate and store task parameters in the constructor; do not run inference there.
3. In `Cases()`, yield one `Case(input=..., workflow=..., metadata=...)` per sample. Put model-visible content in `input`, and reference answers/scoring fields in `metadata`.
4. Use an existing Workflow or implement a new one. A Task must not call a Method directly.
5. Implement `Evaluate()` to score task quality and return numeric values. Define reference-answer normalization and edge-case behavior so they remain consistent across methods.
6. If a new data source is needed, declare it in `config.yaml`/the dataset preparation flow and record its split, revision, and generation parameters. Export the task from `tasks/__init__.py` and select it in `Main.py`.

## Design checks

- The same Task Case should work with different Methods; scoring must not rely on method-private fields.
- Prompt construction, chat templates, and segment boundaries affect reuse results. Define them consistently in the task and record the settings.
- Keep task-quality scoring separate from system-efficiency measurement; do not time inference in `Evaluate()`.
- Task labels and sampling/generation settings should distinguish experimental configurations.
