# KVBench 架构审查报告

## 总体评价

这次重构（`core/Engine.py` → `core/engine/`，`helpers/BenchflowHelper.py` + `helpers/Gpu.py` → `helpers/benchflow/` + `core/engine/Gpu.py`）**是一次净收益明显的改进**：一个上千行的 God Engine 被拆成了 `Engine / Scheduler / GpuGovernor / Reporter / State` 五个职责清晰、各自 100–380 行的协作者，GPU 探测从 `helpers/` 这个杂物间挪到了它真正归属的 `core/engine/`，SkillsBench 的编排代码也从 `helpers/` 顶层下沉成了独立子包。拓扑上，`core.Result`（in-degree 30）/ `core.Config`（18）/ `core.Task`（14）/ `core.Method`（10）构成了一个干净的、无环的抽象内核，方法与任务两侧都是叶子——这是一个健康基准框架该有的形状。但重构只完成了**一半**：拆分出来的协作者仍然通过 `self.engine._gpuPool` / `self.engine._gpuSnapshot` / `self.engine._metrics` 这类私有属性互相穿透，`RunContext` 与 Engine 实例属性形成了"同一份状态、两个家"的分裂；`Engine.Evaluate` 本身仍是一个 ~270 行的巨型方法，说明协作者被抽出来了，编排逻辑却没有真正下放。同时 `helpers/` 依旧承载着后端适配、CacheBlend fork worker、SkillsBench 沙箱三类毫不相干的东西，`helpers/benchflow/sandbox.py`（745 行）和 `helpers/VllmCacheblendPatches.py`（893 行）是新的两个重量级坑。三票对抗验证中**没有任何一条发现被三票一致确认**——这个结果本身值得诚实说出来：本仓库的多数"问题"是有意识的工程权衡（`RunContext` 是显式设计的共享状态；ChatML 字面量是被测试钉住的已知 TODO），而非疏忽。因此下面的评述以"方向性建议"而非"缺陷清单"的口吻给出。

## 1. 依赖图与模块拓扑

```
Main.py
  └── core/                        ← 抽象内核，无外部依赖，无环
        ├── Config.py      (in-deg 18)  ← 叶子，零 intra-import
        ├── Result.py      (in-deg 30)  ← 最重的承重墙
        ├── Task.py        (in-deg 14)  → Result
        ├── Method.py      (in-deg 10)  → Result
        ├── Metrics.py     (in-deg  6)  → Result
        ├── Workload.py    (in-deg  8)  → Result
        ├── Worker.py      (in-deg  3)  → Metrics/Method/Result/Task/Workload
        ├── tui.py         (in-deg  2)  ← 零 intra-import，纯展示
        └── engine/
              ├── State.py     (in-deg 5)  ← RunContext，零 intra-import
              ├── Gpu.py       (in-deg 4)  ← 零 intra-import（重构后正确归位）
              ├── Scheduler.py     → Worker, State
              ├── GpuGovernor.py   → Gpu, State
              ├── Reporter.py      → State
              └── Engine.py        → 上述全部 + core.{Method,Metrics,Task,Worker,tui}

  helpers/                         ← 事实上的三个子系统混居
        ├── ModelAdapter.py (in-deg 9) ← chat 模板 / arch 检测 / parser
        ├── Prompt.py       (in-deg 4) ← 两套复用匹配算法
        ├── VllmHelper.py / TransformersHelper.py / VllmCacheblendPatches.py (893L)
        ├── CacheblendRepoHelper.py (855L) / Qwen3ForCacheBlendRepo.py (in-deg 0)
        ├── SkillInjector.py
        └── benchflow/{helper(456L), sandbox(745L), http_handler, util}

  methods/{Naive, FullPrefill, CacheblendRepo, CacheblendLmcache}
        → core.{Config,Method,Result} + helpers.{Prompt,VllmHelper,...}
  tasks/{Cwe,Niah,Vt,FreshGap,KVCommTasks,Samsum,bases/{KBBase,RulerBase}}
        → core.{Config,Result,Task} + helpers.ModelAdapter
  workload/{RAG, MultiAgentFullConnection, AgentBenchFlow}
        → core.{Result,Workload}
  metrics/{Ttft, Throughput} → core.{Metrics,Result}
  tests/  → 横切全部；test_core_contracts.py 钉住内核契约
```

**承重模块（Hot modules，改动需谨慎）**

| 模块 | in-degree | 角色 |
|---|---|---|
| `core/Result.py` | 30 | 整个仓库的数据货币。任何字段语义变更会波及 methods/tasks/workload/metrics/tests 全线 |
| `core/Config.py` | 18 | 全局 YAML 入口，被 tasks 和 methods 同时依赖 |
| `core/Task.py` | 14 | 任务契约 |
| `core/Method.py` | 10 | 方法契约（KV 复用策略的多态点） |
| `helpers/ModelAdapter.py` | 9 | 唯一的 per-arch chat 模板真理源，被 7 个 task 依赖 |

这五个模块应当被视为**冻结面**：只做加法（新增可选字段/新函数），不做重命名与语义漂移，且每一次改动都应先在 `tests/test_core_contracts.py` 里加断言。

**环与方向性异味**

- 包级**无循环导入**。`core.engine.Engine` 反向 import `core.Method/Task/Worker`，而 `core/__init__.py` 又 re-export `core.engine.Engine`——这是一个"逻辑上的回环"，靠 `State.py` 把 `methods`/`tasks` 字段降级成 `Any` 来规避。可以工作，但类型信息被牺牲了。
- `helpers/benchflow/helper.py` 与 `sandbox.py` 之间存在**对象级回指**（`self._sandbox._helper = self`），这是唯一一处真实的运行期循环引用。
- `helpers/Qwen3ForCacheBlendRepo.py` in-degree = 0：它靠 side-effect 注册（`ModelRegistry.register_model`）被激活，静态图上是孤儿。这是有意为之，但请在文件顶部把"谁在什么时候 import 我"写清楚。

## 2. 单一职责 (SRP)

三票验证没有一致确认任何 SRP 发现，多数被判为"有意的设计"或"已被 docstring 记录的 TODO"。我因此不列"缺陷"，而列**四处规模已经越过舒适阈值、下一次改动会变贵的地方**，按建议强度排序。

1. **`core/engine/Engine.py:88` — `Evaluate` 仍是 ~270 行的单方法主循环**
   - 证据: 同一个 `while True:` 体内混合了 `ctx.eventQueue` 排空、`worker.process.is_alive()` 巡检、初始化/bout/停机三种超时判定、`ctx.fatalError` 更新、新 pair 派发、TUI 刷新与 shutdown 宽限期。
   - 建议: 协作者已经存在，只是编排没下放。把循环体收敛成六行意图声明：
     ```python
     while not ctx.terminated:
         self.scheduler.drainEvents(ctx)
         self.gpuGovernor.tick(ctx, now)
         self.scheduler.reapDeadWorkers(ctx, now)
         self.scheduler.dispatch(ctx)
         self._refreshTui(ctx, now)
     ```
     这是本次重构**唯一未完成的一步**，完成它，重构才算收口。

2. **`core/Worker.py:234` — `WorkerMain` 165 行，三个生命周期阶段共享一个 `try/except/finally`**
   - 证据: `os.setsid()` + `_Redirect` / 初始化 emit / 每-attempt 命令循环 / `finally` 里的 `method.Close()` + `worker_closed`，全部堆在 line 246 的同一个 `try:` 里。
   - 建议: 抽出 `_Initialize(...)` / `_RunOneAttempt(...)` / `_Shutdown(...)`，`WorkerMain` 只剩 spawn 安全的骨架。跨进程边界的代码尤其需要小函数——它的异常路径无法在主进程调试。

3. **`helpers/benchflow/helper.py:29` — `BenchflowHelper` 同时是 HTTP server、沙箱选择器、agent 子进程管理器、watchdog、verifier 执行器**
   - 证据: 11 个实例字段横跨 `_server/_serverThread`、`_sandbox`、`_watchdogThread/_finalResult`、三个输出路径与 `_skillsBlock`；`_WatchdogLoop`(368-425) 一个函数里跑 agent、跑 verifier、读 reward、写 result.json、置 done。
   - 建议: 拆 `AgentSupervisor` / `VerifierSupervisor` / `HttpEndpoint`，`BenchflowHelper` 退化为持有请求队列的 facade。副产品是 `self._sandbox._helper = self`（line 210）这个回指自然消失——改成构造器注入。

4. **`helpers/benchflow/sandbox.py:363` — `ApptainerSandbox` 745 行，含宿主 Python shim 生成**
   - 证据: `_StageHostTools`(684-746) 生成硬编码 `/host_conda/bin/python` 的 shell shim 并 symlink 宿主 site-packages——这是**宿主环境适配**，不是容器管理；`_StageEnvironmentInputs` / `_AgentBinds` / `_ComposeApptainerArgs` 又是另外两类。
   - 建议: `_StageHostTools` 整体外提为 `helpers/benchflow/host_python.py::StageHostPython(...)`；bind 组装降级为模块级纯函数。`ApptainerSandbox` 只保留 SIF pull + prepare + run_agent + run_verifier。

## 3. 耦合与依赖卫生

1. **`core/engine/Reporter.py:148`（及 `GpuGovernor.py:209/214/220`、`Scheduler.py:85`）— 协作者穿透读写 Engine 的私有属性**
   - 证据: `self.engine._gpuPool`、`self.engine._gpuSnapshot = QueryGpus()`、`self.engine._tui.enabled`、`self.engine._metrics`。这些属性**未在 `Engine.__init__` 声明**，而是在 `Evaluate` 中途（line 97/101/116）赋值。
   - 建议: 反方票有理——`_metrics` 是不可变输入、`_tui` 是 Engine 自有单例，把它们塞进 `RunContext` 反而更糟。真正该动的是 **`_gpuSnapshot`**（被 GpuGovernor 写、被 Reporter 读，且 `gpuSnapshotAt` / `gpuSnapshotError` 已经在 `RunContext` 上——同一概念被劈成两半）。把 `gpuSnapshot` 挪到 `RunContext`，其余两个改成显式只读访问器（`engine.isTuiEnabled()`），足矣。这是**中低优先级的整洁性问题，不是缺陷**。

2. **`core/__init__.py:9` — 顶层 barrel re-export 了子包 `core.engine` 的 `Engine` 与两个异常类**
   - 证据: `from .engine import (BenchmarkInitializationError, BenchmarkResourceReleaseError, Engine)`，而 `core/engine/Engine.py:22-25` 又 `from ..Method import Method` 反向依赖。
   - 建议: `core/__init__.py` 只暴露抽象内核（Config/Metrics/Method/Result/Task/Workload），让 `Main.py` 写 `from core.engine import Engine`。收益是 engine 子包未来的再拆分变成局部改动。

3. **`core/engine/State.py:182` — `methods` / `tasks` 被标注为 `Any` 以规避循环 import**
   - 证据: 类型信息在最需要它的编排层丢失。
   - 建议: 用 `typing.Protocol` 在 `core/Method.py` 定义 `MethodLike`，或 `if TYPE_CHECKING:` + 前向引用。一行级修复，收益是 IDE 与 mypy 重新看得见调度层。

4. **`helpers/benchflow/helper.py:210` — `self._sandbox._helper = self` 回指**
   - 证据: 已在 §2 引用。这是全仓库唯一的运行期循环引用，且穿透了 sandbox 的私有名。
   - 建议: 构造 `Sandbox` 时把它需要的东西（outputDir、请求队列、skills block）作为参数传入。

## 4. 内聚性

1. **`helpers/__init__.py:1` — docstring 声称 "one module per backend, no grab-bag"，实际是四个子系统混居**
   - 证据: (a) 后端适配 `VllmHelper/TransformersHelper/VllmCacheblendPatches/ModelAdapter`；(b) SkillsBench 编排 `SkillInjector + benchflow/*`；(c) CacheBlend fork 的 JSONL worker `CacheblendRepoHelper`(855L) + OOT 模型类 `Qwen3ForCacheBlendRepo`；(d) prompt 复用算法 `Prompt.py`。四者零共享抽象。
   - 建议: `helpers/backends/`（vllm/transformers/patches/ModelAdapter/Prompt）、`helpers/cacheblend_repo/`（RepoHelper + Qwen3 OOT）、`helpers/benchflow/`（已存在）。三个子包，三条独立演进线。docstring 随之改写为真实范围。

2. **`helpers/ModelAdapter.py:1` — 355 行里塞了 7 个关注点**
   - 证据: arch 检测(41-98)、tokenizer 惰性加载(106-116)、thinking kwargs 翻译(124-135)、message 规范化含 tool-call JSON 重塑(138-191)、chat 渲染(194-229)、边界串提取(237-276)、parser 分发(296-356)。其中 `vars(fn) if hasattr(fn, "__dict__") else dict(fn)` 直接掏对象内部；`from time import time_ns` 被塞进按调用执行的路径。
   - 建议: 拆 `helpers/chat_template.py`（arch/tokenizer/render/boundaries/thinking）与 `helpers/parsers.py`（`_ARCH_PARSERS` + tool-call 重塑）。**注意它 in-degree = 9**，拆分必须保持 `from helpers.ModelAdapter import ...` 的旧入口作为转发层，分两步走。

3. **`core/tui.py:19` — `BenchmarkTui` 311 行混合 Rich 渲染、termios 原始模式、输入线程、六个视图**
   - 证据: `_Render/_Core/_Full/_Timing/_Schedule/_Logs/_Failures` 七个渲染方法 + `Start` 里的 cbreak 切换与 `Stop` 里的 termios 还原 + `_ReadKeys/_HandleKey` 自带线程 + `_stop/_quit/_thread/_termAttrs/_inputEnabled` 五个私有标志。`_Render` 用局部 dict 分发视图，加一个视图要改两处。
   - 建议: TUI 还年轻，现在拆代价最低：`TerminalMode`（termios 获取/还原，contextmanager）、`KeyReader`（线程 + 键位映射）、`DashboardView`（纯渲染，输入是 snapshot）。三者各有一条独立的变更原因。

4. **`core/Worker.py:17` — 模块里混入了与 worker 无关的报表聚合助手**
   - 证据: `_NormalizeScores`(17-22) 与 `_AggregateScores`(25-29) 是纯报表整形，module docstring 自称 "Spawn-safe method worker and the single-pair evaluation loop"，两者都不在其中。
   - 建议: 移入 `core/Result.py`。`_Redirect` / `_Emit` 留在原地是合理的。

5. **`helpers/Prompt.py:1` — 两套零共享代码的匹配算法同居**
   - 证据: `ComposeReuse`(20-129) 是排列 DFS，返回 `Tuple[Optional[List[str]], str]`；`ComposeInterleavedReuse`(132-397) 是 Aho-Corasick + 加权区间 DP，返回 `List[Tuple[Optional[int], str]]`。
   - 建议: 优先级低（有 `tests/test_prompt.py` 护着，且改动频率低）。若要拆，按算法拆成两个模块即可。

## 5. 可扩展性与设计品味

**对本次重构的诚实判词**：方向完全正确，执行完成了约 70%。

**做对了的**：
- `core/engine/` 的五文件切分是教科书式的——`State.py` / `Gpu.py` 零 intra-import 处在叶子位，`Scheduler`/`GpuGovernor`/`Reporter` 各自只依赖 `State` 加一个具体依赖，扇入扇出都收敛。
- `helpers/Gpu.py` → `core/engine/Gpu.py` 是纯粹的归位：GPU 池调度是 engine 的领域知识，从来就不该在 helpers。
- `helpers/benchflow/` 子包化让 SkillsBench 这条重资产线（1300+ 行）有了独立边界。

**回退了的 / 未收口的**：
- `Engine.Evaluate` 没有随协作者的诞生而瘦身，反而成了新的"胶水巨兽"——协作者存在但不自治，Engine 仍在替它们做决策。
- 引入 `RunContext`（35 字段）后没有把 Engine 实例上的 `_gpuPool`/`_gpuSnapshot` 一并迁入，造成**同一概念的状态分居两处**（`ctx.gpuSnapshotAt` 在 context 上，`engine._gpuSnapshot` 在实例上）。这是重构半途而废最典型的残留物。关于 `RunContext` 本身是否"过大"：三票中两票认为它是**有意的、无方法的共享状态容器**，我同意——不要为了字段数去拆它，只要把散落在 Engine 实例上的孤儿字段收进来即可。
- `helpers/` 的整理只做了 benchflow 一条线，CacheBlend fork 那条线（855 + 241 + 893 行）原地未动。

**扩展性上的具体阻力点**：

1. **`workload/MultiAgentFullConnectionWorkload.py:54` — 手写 Qwen3 ChatML 字面量，绕开 `ModelAdapter.render_chat`**
   - 证据: `f"<|im_start|>system\n{...}<|im_end|>\n...<|im_start|>assistant\n<think>\n\n</think>\n\n"`。同文件 line 50-53 的 docstring 已自认应由 `ModelAdapter.render_chat` 负责 per-arch kwargs。
   - 分歧与结论: 三票分裂。反方有力的理由是——`MultiAgentFullConnectionInput.chatTemplate: bool = True` 已提供 opt-out，`tests/test_multiagent.py:21-22` 把该字面量钉成了契约，且当前默认 Qwen3 下与 `render_chat` 逐字节等价，**今天没有可观察的回归**。但仓库刚刚接入 Muse Glimmer（`<|start|>/<|message|>/<|eot|>`）与 Mistral（`[INST]`），一旦有人用非 Qwen 模型跑 `tasks/KVCommTasks.py` 的四个 KVComm 任务，得到的是训练时从未见过的 role tag——静默的分数崩塌，不是崩溃。**我把它定为严重度 3（而非原报告的 5）：不是缺陷，是一颗已上膛的扩展性地雷。**
   - 建议: 给 `MultiAgentFullConnectionInput` 加 `modelPath` 字段，`_BuildPrompt` 改调 `render_chat`；同时把 `test_multiagent.py` 的断言从"字面量相等"改为"与 `render_chat` 输出相等"，这样测试就从**锁死**变成**守护**。

2. **`helpers/Qwen3ForCacheBlendRepo.py` in-degree = 0，靠 side-effect 注册生效**
   - 证据: 静态依赖图上是孤儿，实际通过 `AutoConfig.register` + `ModelRegistry.register_model` 在 import 时挂载。
   - 建议: 机制本身是对的（OOT 模型只能这么做），但请在模块顶部用 5 行注释写明"由谁 import、在何时、为什么不能显式引用"。否则下一个人会把它当死代码删掉。

3. **`config.yaml` 单文件承载数据布局 / 模型选择 / CacheBlend fork 路径 / 50 行 SkillsBench 沙箱配置**
   - 证据: `DatasetPath`(21)、`ModelPath`(24)、`Cacheblend.Repo.RepoPath`(34)、`AgentBenchFlow`(38-92)。换数据集和换沙箱后端改的是同一个文件。
   - 建议: `core.Config.LoadConfig` 已支持 `path` 参数，让它 merge 多份：`datasets.yaml` / `models.yaml` / `benchflow.yaml`。低成本，收益是每份配置只有一个变更理由。

## 6. 速胜 vs 深层重构

| 发现 | 分类 |
|---|---|
| `core/engine/State.py:182` — `methods`/`tasks` 的 `Any` 改为 `Protocol` / `TYPE_CHECKING` | Quick win (< 30 min) |
| `core/Worker.py:17` — `_NormalizeScores`/`_AggregateScores` 移入 `core/Result.py` | Quick win (< 30 min) |
| `core/__init__.py:9` — 移除 `Engine` 的顶层 re-export，`Main.py` 改用 `core.engine` | Quick win (< 30 min) |
| `core/engine/*` — `_gpuSnapshot` 迁入 `RunContext`；`_tui` 改为 `isTuiEnabled()` 访问器 | Quick win (< 30 min) |
| `helpers/ModelAdapter.py:296` — `from time import time_ns` 提到模块顶层 | Quick win (< 30 min) |
| `helpers/__init__.py:1` — 改写 docstring 使其描述真实范围 | Quick win (< 30 min) |
| `helpers/Qwen3ForCacheBlendRepo.py` — 补 side-effect 注册的说明注释 | Quick win (< 30 min) |
| `config.yaml` — 拆成 datasets / models / benchflow 三份并 merge | Quick win (< 30 min) |
| `workload/MultiAgentFullConnectionWorkload.py:54` — 改调 `render_chat` + 改写测试断言 | Deeper refactor (> 1 hour) |
| `core/engine/Engine.py:88` — `Evaluate` 拆成六步编排，逻辑下放给协作者 | Deeper refactor (> 1 hour) |
| `core/Worker.py:234` — `WorkerMain` 拆 `_Initialize`/`_RunOneAttempt`/`_Shutdown` | Deeper refactor (> 1 hour) |
| `core/tui.py:19` — 拆 `TerminalMode` / `KeyReader` / `DashboardView` | Deeper refactor (> 1 hour) |
| `helpers/benchflow/helper.py:29` — 拆 `AgentSupervisor`/`VerifierSupervisor`/`HttpEndpoint`，消除回指 | Deeper refactor (> 1 hour) |
| `helpers/benchflow/sandbox.py:363` — 外提 `host_python.py`，bind 组装函数化 | Deeper refactor (> 1 hour) |
| `helpers/` — 拆 `backends/` + `cacheblend_repo/` 两个子包 | Deeper refactor (> 1 hour) |
| `helpers/ModelAdapter.py:1` — 拆 `chat_template.py` + `parsers.py`（需保留转发层） | Deeper refactor (> 1 hour) |

## 7. 下一步建议 (Top 7)

1. **收口 engine 重构**：把 `Evaluate` 的循环体下放给已有协作者，目标是主循环 ≤ 15 行。这是当前架构上**唯一的高杠杆动作**——不做完，`core/engine/` 的拆分只兑现了一半价值。
2. **消灭状态分居**：`_gpuSnapshot` 迁入 `RunContext`（`gpuSnapshotAt`/`gpuSnapshotError` 已在那儿）；`_tui`/`_metrics` 保留在 Engine 但改为显式访问器。同时在 `Engine.__init__` 里声明所有实例属性，别再在 `Evaluate` 中途长出来。
3. **拆 `WorkerMain`**：跨进程边界的代码调试成本最高，三个生命周期阶段必须是三个可单测的函数。`tests/test_worker_pair.py` 已有基础，拆完立刻补测。
4. **`helpers/` 三分**：`backends/` + `cacheblend_repo/` + `benchflow/`。先只移动文件、保留旧路径的转发 import，一次 commit；下一次 commit 再删转发。`ModelAdapter` in-degree = 9，必须走两步。
5. **修 `MultiAgentFullConnectionWorkload` 的 ChatML 字面量**，并把 `test_multiagent.py` 的断言改成"与 `render_chat` 等价"。在 Muse Glimmer / Mistral 已进仓库的今天，这是唯一会造成**静默错误结果**的扩展性缺口。
6. **给承重模块加护栏**：在 `tests/test_core_contracts.py` 里把 `Result` 的字段集、`Method`/`Task` 的抽象方法签名、`Config` 的必需键显式断言一遍。in-degree 30 的模块值得一份"改动即报警"的合同测试。
7. **拆 TUI**（趁它还只有 311 行）：`TerminalMode`/`KeyReader`/`DashboardView`。同时把 `_Render` 的视图分发 dict 提为模块级注册表，让"加一个视图"只需改一处。

## 8. 已经做得好的地方

这些不是客套，是这个仓库真实的、不容易达到的成果：

- **`core/` 抽象内核干净且无环**。`Config`/`Result`/`State`/`Gpu`/`tui` 全部零 intra-import，`Task`/`Method`/`Metrics`/`Workload` 各自只依赖 `Result`。一个跑真实 GPU 负载的基准框架能保持这种拓扑纯度，说明抽象是在写代码之前想清楚的。
- **Method 动物园是完整的**：Naive / FullPrefill / CacheblendRepo / CacheblendLmcache 四条互相独立的实现，共享同一个 `core.Method` 契约。这正是基准框架该有的可替换性——加第五种方法不需要碰任何既有代码。
- **真实的、跑通的硬工程**。vLLM 0.25 + LMCache 的进程内 blending（含模块 import guard 打进 EngineCore 子进程、三处运行期 patch、首次 store 损坏的 warmup 修复、prefix-blend + tail-recompute 的 partial-retrieve 修复并 64/64 验证）；官方 CacheBlend fork（vLLM 0.4.1）上以纯加法方式支持 Qwen3-8B（`AutoConfig.register` + OOT 模型类 + 绕过 fork 的 3-D RMSNorm CUDA kernel），且验证 fuse == full == stock 同时 Mistral 无回归。这类跨版本、跨 fork 的兼容工作没有捷径。
- **多 GPU 与大模型路径已打通**：Qwen3-32B 在 2×L40 TP=2 上 ~98% reuse；Muse Glimmer 30B 在 TP=2 上跑通 FullPrefillVllm，并为其在 `TemplateHelper` 里做了 Meta 风格 chat 格式的 arch 检测。
- **SkillsBench 集成 6/6 跑通**，并且诚实地把 reward=0 定位到 agent 内容质量而非 skill 注入机制——能区分"我的基础设施坏了"和"被测对象就是这个水平"，是基准作者最重要的素质。
- **`tests/` 层次分明**：`test_core_contracts.py` 守内核契约，`test_worker_pair.py` 守跨进程路径，`test_engine.py`/`test_gpu.py`/`test_tui.py` 守 engine 子包，`test_prompt.py` 守两套复用算法，`test_cacheblend_repo_{gpu_buffers,lifecycle}.py` 守最脆弱的 fork 集成。11 个测试文件对 11k 行代码，覆盖的是正确的地方。
- **`helpers/ModelAdapter.py` 作为唯一 chat 模板真理源**这个决策是对的——7 个 task 家族已经正确委托给它。§5 指出的 workload 字面量恰恰是唯一的例外，反过来证明了这条抽象的价值。
- **文档化的 TODO 文化**：多处（`MultiAgentFullConnectionWorkload.py:50-53`、`State.py` 的 docstring）在代码里明确写下了"当前实现与目标设计的差距"。这比沉默的技术债好一个数量级。
