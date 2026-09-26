# Workflow Interface and Existing Workflows

A Workflow defines the order of requests and how reusable state is shared across them. It does not call a Method directly. Instead, it returns Actions for the Engine, which invokes `Method.Prepare` or `Method.Run` and sends the results back to the Workflow.

## Protocol

The main types in `core.Workflow` are:

- `ActionKind.PREPARE` / `ActionKind.RUN`: distinguish cache preparation from inference.
- `Action(kind, case_id, data, tag="", retainOutput=False)`: one request action. PREPARE `data` is a `list[str]`; RUN `data` is a complete prompt string. `tag` identifies an agent or step. `retainOutput` hints that a later request may reuse this generated state.
- `ActionResult(case_id, result, tag="")`: result returned by the Engine after executing an Action.
- `Workflow.next()`: returns the next list of Actions, or `None` when the workflow is complete.
- `Workflow.observe(results)`: receives results from the previous step and updates workflow state.
- `Workflow.finished`: indicates whether the workflow has completed.

Every call to `next()` must return Actions of a single kind; PREPARE and RUN actions cannot be mixed in the same step. A step may return multiple Actions of that kind for batched execution. Each `Case.workflow` owns the state for its Case.

## Existing workflows

### `RAGWorkflow`

Driven by `RAGInput(prepare_input, run_input)`. If `prepare_input` is non-empty, it issues one PREPARE followed by one RUN. Otherwise it issues only a RUN. This workflow is used for static requests such as RULER, knowledge-base QA, and summarization.

```python
data = RAGInput(prepare_input=[document_a, document_b], run_input=full_prompt)
workflow = RAGWorkflow(case_id=case_index, data=data)
```

### `MultiAgentFullConnectionWorkflow`

Represents a fixed fully connected multi-agent conversation. It first prepares the shared task text, then runs agents in sequence; each agent's prompt may include outputs from earlier agents. An optional decision agent runs last to produce a final answer. `retainOutput` indicates whether a RUN result may be reused by a later agent.

### `AgentBenchFlowWorkflow`

Connects a real BenchFlow/SkillsBench rollout to KVBench. It starts the agent runner/provider endpoint, receives the agent's model requests, and converts them into KVBench actions. Skill documents can be prepared separately and reused in model requests. Depending on configuration, it can run the full rollout or collect metrics only for the first model RUN, then map the official BenchFlow result to the final Task result.

## Adding a workflow

When a task's request order or state-sharing rules cannot be expressed by an existing workflow, implement `Workflow` under `workflow/`. Store the current step, generate same-kind Actions in `next()`, consume results and advance state in `observe()`, and keep the semantics of `finished` consistent with `next()`. Attach the workflow through `Case.workflow`; neither the Task nor Workflow should call a Method directly.
