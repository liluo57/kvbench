# Qwen3.8-27B CacheRepair Stage A 报告

日期：2026-10-22。范围为用户确认的训练准备 + Stage A 工程烟测。未运行 Stage B/C，未接入新的在线 Method。

## 结果

真实本地 Qwen3.8-27B BF16 权重、GDN + full-attention 前向，以及现有 54,017,664 参数 repairer 的训练、保存、重载路径均已通过。
32 条训练 / 8 条验证，1 epoch、effective batch=4、8 个 optimizer steps。统计量来自固定的 8 条训练样本。

| 固定数据集上的 normalized residual MSE | 零 residual 基线 | 训练后 |
| --- | ---: | ---: |
| 训练 32 条 | 1.133569 | 1.052218 |
| 验证 8 条 | 0.950402 | 0.885177 |

验证 MSE 较零 residual 下降 6.86%。原始 residual relative L2 error=0.953873，预测/真实 residual L2=0.299862，预测 residual/stale KV L2=0.096060。
这些数值仅证明小规模训练通路产生了非零且可降低所测 KV residual error 的修正；没有回答质量、泛化能力或 TTFT 提升结论。不同 batch 的 loss 不应直接比较为学习曲线。

## 数据与去重

固定六类 CoRAG train 来源，每类前 16 条，共 96 条候选 query、574 个 passages。保留检索顺序，从干净 passages 构造 2..4 chunks，预算上限 256 tokens。不是官方 50,000 条训练集的复现。
评测输入清单覆盖 243,608 段不同内容、146,851,665 个 Qwen tokens。

| 筛查类型 | 确认匹配的独立 passages |
| --- | ---: |
| 归一化完整 passage exact duplicate | 0 |
| 经实际 token 验证的 50-token 重叠 | 20 |
| 至少 12 词的完整 evidence sentence | 43 |
| 以上污染集合的并集，实际排除 | 47 |

各类型和各 workload 之间有重复，不能相加。匹配来自 MuSiQue 6、HotpotQA 10、TriviaQA 34 个 passages。
MinHash/LSH 产生 5 个匹配事件、4 个候选 passages；它们均已因确认重叠排除，待审查队列为 0。另有 1 个 NQ query 因不足两个干净 passages 被丢弃。

全部本地静态 benchmark 文件均扫描。AgentBenchFlow 仅覆盖 task.md/SKILL.md，动态工具输出、任务资源和生成对话未完整筛查，状态为 partial，不能宣称这部分零污染。Copy/FreshGap 覆盖默认参数、seed=42、前 100 cases，自定义配置需重新 inventory。

| 来源 | 训练 | 验证 | 排除 passages |
| --- | ---: | ---: | ---: |
| eli5 | 4 | 2 | 4 |
| fever | 6 | 1 | 14 |
| nq | 7 | 0 | 11 |
| wow | 4 | 2 | 16 |
| trex | 7 | 2 | 1 |
| structured_zeroshot | 4 | 1 | 1 |

通过完整 Wikipedia 文档 ID、内容哈希、共享 spans 和近重复候选的关联组划分，训练和验证的源文档互不交叉。验证集不含 NQ，不能用于来源级泛化判断。
训练共 5423 tokens，85..247 tokens/example；验证共 1317 tokens，111..216 tokens/example。训练 chunk counts：{'2': 12, '4': 9, '3': 11}；验证：{'2': 4, '3': 4}。

## 版本与适配

CR commit：`4eb96cd92df4be4f679ad8ac42a51442249683c2`。官方 train.py/model.py/cache.py 原样保留，与独立 checkout 逐字节一致；上游源码 diff 为零。
官方 train.py SHA256：`1698761ec29164c501c0421b249c18c952538a997cb44e0064391d679e29946a`。

原训练循环、RMS、normalized residual MSE、AdamW、clipping、shuffle 和 LR schedule 直接执行官方函数代码；只通过私有依赖绑定替换目标 loader、geometry、make_pair、现有 Qwen repairer 和 checkpoint writer。
目标使用完整真实 mixed forward；GDN states 不进入 repairer 输入、输出或 loss。跨 GPU packing 是现有 cache.py 的唯一功能扩展，默认收集到第一个 attention layer 的 device，也可显式指定 repairer device。

| 新增/修改文件 | 用途 |
| --- | --- |
| supervision.py / cache.py | 正常 native stale/joint forward 与跨卡 attention KV packing |
| sources.py / eval_inventory.py | 固定 revision range reads、缓存、tokenizer/目标 provenance、本地评测内容清单 |
| decontam.py / prepare.py | 分层去重、去重缓存、源文档隔离 split、token JSONL/manifest/report |
| train.py / configs/Qwen3.8-27B-StageA.json | 官方循环薄适配、梯度与 checkpoint 诊断、可配置 Stage A |
| tests/*/test_cacherepair_qwen38_{data,training}.py | schema、重叠、split、RMS、native pairs、累积、disk cache、重载 |

CoRAG query revision：`d3f3f906826b1524e7db4301723cecad770be0a7`。Passage revision：`bf95d9b4b19c68420395efad7498d0ef5052ab4b`。
本地 ModelScope metadata 声明全部权重 shards 的 revision 为 `1098534ab5d7220ea0f4a6b9f07bb03729a79c1d`。Tokenizer JSON 同 revision；tokenizer config 为 `f17d3260b0dac5e741690c873199c757d74d2ff2`；chat template 为 `a23c845fc2aa3b7f3979e32397cea12a57551631`。
这些是本地下载记录声明，未重新哈希所有权重。逐文件记录见 target-provenance.json；tokenizer 的实际文件内容 revision 和哈希另存于 dataset.json。训练结束后补读的 provenance 已作为 sidecar 加入根目录与 checkpoint。
环境：隔离 Transformers 5.8.0、Torch 2.13.0+cu129、HF Hub 1.33.0。没有改变主项目 Transformers 4.57.6；与官方 Torch 2.6 的逐位数值复现不作等同。

## 工程与资源

| 项目 | 测量值 |
| --- | ---: |
| 模型加载 | 63.99 s |
| 8 个训练样本的 RMS 估计 | 12.53 s |
| 训练更新，不含独立统计与验证 | 35.99 s |
| 训练吞吐 | 0.889 examples/s; 150.68 tokens/s |
| 统计、训练、验证与最终训练集诊断合计 | 60.25 s |
| GPU 1 / GPU 2 peak allocated | 26.245 / 25.190 GiB |
| Checkpoint | 206.317 MiB |
| Repairer trainable parameters | 54,017,664 |

40 个唯一 pairs 按需生成，41 次后续访问复用 run-local disk cache；不在 GPU 保存整个数据集。每个 RMS buffer 为 [16,4,2,256]、有限且为正。Checkpoint 含 70 个 tensors，不含目标 embedding。
8 步均有有限非零 repairer 梯度；第 1 步非零梯度仅输出头，第 2 步 32 组、第 3..8 步全部 68 组。Target 全部冻结、无梯度、未进入 optimizer；抽样权重指纹训练前后一致。重载模型输出逐元素一致。
专项测试 32 passed；相关既有回归 72 passed（连同较早的专项部分在 90 项命令中通过）。

## 逐层验证误差

下表为按 token 加权的 normalized coordinate MSE；前面的总体 MSE 按 example 平均，因此两者的权重不同。

| Native layer | K | V |
| ---: | ---: | ---: |
| 3 | 0.806336 | 0.841074 |
| 7 | 0.833675 | 0.873076 |
| 11 | 0.880285 | 0.950138 |
| 15 | 0.891927 | 0.973837 |
| 19 | 0.871687 | 0.961389 |
| 23 | 0.882647 | 0.896797 |
| 27 | 0.837090 | 0.904618 |
| 31 | 0.838909 | 0.939185 |
| 35 | 0.803325 | 0.926775 |
| 39 | 0.773466 | 0.784232 |
| 43 | 0.756198 | 0.781129 |
| 47 | 0.748574 | 0.785885 |
| 51 | 0.750371 | 0.799665 |
| 55 | 0.735053 | 0.757105 |
| 59 | 0.749070 | 0.791186 |
| 63 | 0.752241 | 0.738514 |

## 产物与复现

工作区产物：

- `outputs/cacherepair-qwen38/stage-a-data-v2/`：train.jsonl、validation.jsonl、dataset.json、examples.manifest.jsonl、deduplication.json/md。
- `outputs/cacherepair-qwen38/stage-a-training-v1/checkpoint-8/`：模型、RMS buffers、配置、target-provenance.json。
- `outputs/cacherepair-qwen38/stage-a-training-v1/`：loss.jsonl、statistics.safetensors、diagnostics.json、stage-a-summary.json、target-provenance.json。
- `.cache/cacherepair-qwen38-evaluation-v1/`：可复用评测内容 SQLite、数据源文件清单与覆盖状态。
- `.cache/cacherepair-qwen38-sources/`：固定源 revision 的 metadata/passages 与去重结果缓存。

具体命令、接口与适配映射见 [TRAINING.md](TRAINING.md)。所有实验目录均新建，较早的数据 v1 保留；未覆盖既有 benchmark 输出。

下一阶段若另行开展，应扩大并平衡来源/长度与验证集，重新估计统计量，再完成 Hypic Addition/TRR + repairer 在线路径和实际质量/TTFT 对照。当前 checkpoint 是 Stage A 烟测产物。
