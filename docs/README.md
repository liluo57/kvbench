# KVBench Documentation

The interface guides in this directory describe the KVBench framework independently of the paper's experiments. The paper reproduction procedure is documented separately in [Reproduction.md](Reproduction.md).

## Documentation

- [Method interface and extension](Method.md): method lifecycle, inputs and outputs, existing methods, and how to add a method.
- [Task interface and extension](Task.md): Cases, datasets, scoring, and how to add a task.
- [Workflow interface and current workflows](Workflow.md): the Action protocol and the RAG, multi-agent, and BenchFlow workflows.
- [BenchFlow integration](AgentBenchFlow.md): SkillsBench sources, runtime configuration, and remote Docker setup.
- [Paper reproduction guide](Reproduction.md): dataset preparation, model and method configuration, experiment runs, and metrics.

## Entry point

The repository's `Main.py` constructs the task, method, and metric lists, then calls `core.engine.Engine.Evaluate()`. Before running it, install the dependencies and configure `config.yaml` as described in the root [readme.md](../readme.md). Public datasets are not checked into git; prepare them from the configuration:

```bash
python scripts/PrepareDataset.py
python Main.py
```

The Engine assigns GPUs according to the `Engine` configuration, launches isolated workers, and schedules each method–task pair. Each run creates an output directory under `Engine.OutputRoot` containing a manifest, event log, and aggregate results. `Main.py` is the current checkout's example entry point; reproducing the complete paper matrix requires changing the model, task, and method lists as described in the [reproduction guide](Reproduction.md).
