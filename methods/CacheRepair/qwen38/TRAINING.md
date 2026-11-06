# Qwen3.8 训练准备与 pilot

全量训练、30 分钟 GPU 排队和恢复命令见 [FULL_TRAINING.md](FULL_TRAINING.md)。

已完成 32 train + 8 validation 的 Stage A 工程烟测。后续训练采用 Stage B：
256 train + 32 validation、每例最多 1,024 tokens、一个 epoch。
模型结构继续使用现有 `qwen38` repairer。后续增加的在线 Hypic TRR + Repairer
Method 与质量/TTFT 评测入口见 [HybridCacheRepair](../../HybridCacheRepair/README.md)。

已执行的 Stage A 数据、指标、资源与产物见 [STAGE_A_REPORT.md](STAGE_A_REPORT.md)。
已完成的 Stage B 结果见 [STAGE_B_REPORT.md](STAGE_B_REPORT.md)，
配置见 `configs/Qwen3.8-27B-StageB.json`。

## 官方实现与适配

使用 [CacheRepair-code](https://github.com/genglinWang/CacheRepair-code/tree/4eb96cd92df4be4f679ad8ac42a51442249683c2)
commit `4eb96cd92df4be4f679ad8ac42a51442249683c2`。
已将该 commit checkout 到 `.cache/cacherepair-upstream`，并与 vendored 文件逐字节比较。
`backend/cacherepair/train.py`、`model.py`、`cache.py` 均保持原样；**上游源码 diff 为零**。

`qwen38/train.py` 通过 Python 函数的私有 globals 绑定执行官方代码，不修改官方模块
的全局变量，也不引入第二套优化循环。后续恢复功能对该函数做经过结构断言的 AST
适配，只增加状态读写与起始 epoch/batch；原始 vendored 文件仍不修改：

| 官方内容 | 本地处理 |
| --- | --- |
| `train.main` 中的 epoch、shuffle、variable-length 梯度累积和日志 | 原函数代码直接执行 |
| `compute_stats` | 原函数直接执行；可显式限制为训练集中的固定子集，默认完整训练集 |
| `CacheRepair.loss` | 原函数直接作为现有模型的 loss，normalized residual MSE |
| `learning_rate` | 原函数，2% warmup 与 peak/min cosine；两个 pilot 的 horizon 均为 1 epoch |
| AdamW / clipping | 原循环调用，betas=(0.9,0.95)、eps=1e-8、weight decay=0.01、clip=1.0 |
| batch size | 原循环固定 effective batch=4；尾批按实际例数平均 |
| generic target loader | 替换成原生 Qwen3.5 wrapper，读取实际 Qwen3.8 权重，允许按 decoder layers 分 GPU |
| `TargetConfig` | 使用现有 `TargetGeometry`，只映射 full-attention layers |
| generic `CacheRepair` | 替换为现有 `QwenCacheRepair`，参数和网络结构不变 |
| generic `make_pair` | `supervision.py` 执行独立 chunk 与完整 joint forward，GDN 正常计算 |
| generic checkpoint writer | 使用现有严格兼容性 writer，附加官方训练 metadata、诊断和重载验证 |

官方硬编码的 target `.to("cuda")` 在薄 facade 中表示“使用已 dispatch 的原生目标”。
Repairer `.to("cuda")` 则映射到目标 embedding 所在 device。否则 27B BF16 会被错误地
整体搬回单卡。跨卡 attention cache 在 packing 时汇集至 repairer device，写回时仍
由原接口按 native layer device 转回。这些适配只影响新训练入口。

RMS 仍按全部训练 token 加权，FP64 累加、clamp mean square 到 1e-12 后开方为 FP32。
训练按 example 平均 loss，逐例 forward/backward，因此没有 padding token 进入 loss。
目标 BF16、canonical KV / repairer / RMS FP32。每次 optimizer step 检查有限梯度、
repairer 非零梯度、target 无梯度及 optimizer 与 target 参数隔离。
每个 epoch 保存 checkpoint 并检查重载输出逐元素一致。

## 数据来源：可恢复部分与 pilot 策略

[论文附录 J](https://arxiv.org/html/2609.35139v1#A10) 给出六个 CoRAG/KILT 来源和
first-k retrieval 顺序（k=2..20），以及 50,000 条训练数据的长度分布和来源配比。
官方代码只提供接收 token chunks 的训练入口；没有发布那 50,000 条 query 的完整
选样 manifest、原始 revision 与抽样实现。因此本流程是明确标注的 pilot 近似，不能
声称精确复现论文训练集。

固定来源与获取时核对的 schema：

| 资源 | Revision | Schema / 获取范围 |
| --- | --- | --- |
| `corag/kilt` | `d3f3f906826b1524e7db4301723cecad770be0a7` | `query, query_id, answers, context_doc_ids`; 六来源的 train |
| `corag/kilt-corpus` | `bf95d9b4b19c68420395efad7498d0ef5052ab4b` | `id, contents, title, wikipedia_id`; train |

来源为 `eli5, fever, nq, wow, trex, structured_zeroshot`。Hub metadata 标注二者为
非 gated、Apache-2.0；原资源及归属参见各自 [query card](https://huggingface.co/datasets/corag/kilt)
和 [passage card](https://huggingface.co/datasets/corag/kilt-corpus)。六来源 query 文件全量
约 892 MB，passage corpus 为 37 shards、约 10.15 GB。Inventory 命令只读 metadata。

Pilot 先取每个来源 train 的前 N 条记录，默认 N=16，总计 96 queries；每条取前 6 个
已发布 retrieval IDs，保留顺序。此 bounded prefix sample 存在顺序偏差，专供工程检查。
`HfFileSystem` + PyArrow 读取所需 row groups（每组 1000 行），不下载全部 corpus。
真实 passage ID 必须等于其分片累计 row offset，否则报错；不会默默取错 passage。
已获取的 metadata/passages 与最终候选记录写入 `.cache`，重试复用。
Corpus range reads 使用 spawn 进程池隔离每个 worker 的 HTTP client，避免某一连接
重试时 HF Hub 关闭共享客户端、连带中断其他线程的读取。Query/metadata 的小批读取
继续使用线程池。该适配不修改全局 HF 环境或依赖源码。

去重后，按 query ID、Wikipedia 文档 ID、内容哈希、共享 token spans 和近重复候选
构建关联组，整组分配 train/validation，然后才构造 chunk。每个例子保留 2..4 个
有序 passages，长度在上限内确定性变化。源 passage 的完整内容先筛查，随后才截断。
若污染 passage 被移除且剩余少于两个 chunks，丢弃该 query。
近重复候选保留可审查分数，未审查前隔离于本次 pilot，不以弱相似度直接删除。

格式复用 KVBench `QABase` 的 `title + "\n\n" + body + "\n\n"`，第一 chunk 使用
现有 `MusiqueTask.prefixPrompt`。每块独立编码，`add_special_tokens=False`；不额外
添加 ChatML。截断保持末尾分隔 token。Hypic 的 system/tail priming 是离线缓存暖身，
不属于传给 PREPARE 的文档 token；训练 pair 使用原始 chunks 独立初始化 GDN，
不把暖身或其他 chunks 的 GDN 状态混入 stale branch。
Source query/answers 只记录为 provenance，不进入 repairer 输入。

每个 example manifest 记录源 ID、revision、retrieval rank、Wikipedia ID、内容指纹、
截断标志、token/chunk 数、随机 seed、split 和 tokenizer 内容 revision。
本地 tokenizer 使用文件内容哈希作为 local revision。目标 checkpoint 的 ModelScope
`.msc` 下载记录含逐文件 revision，另存为 provenance；它属于下载记录声明，没有
重新逐字节哈希全部权重。config/index 文件的哈希也单独保存，不将它们称为权重哈希。

## 去重与覆盖范围

`evaluation-inventory` 逐行读取仓库实际本地输入，建立去重后的 SQLite 文本清单及
引用表，记录 source 文件哈希：MuSiQue、2WikiMultiHopQA（WikiMQA）、HotpotQA、
TriviaQA、SAMSum、MultiNews、GovReport、MMLU test、GSM8K、HumanEval、全部本地
RULER NIAH/CWE/VT 文件，以及默认 seed 下的 Copy/FreshGap。
RULER shuffle variants 使用相同 source 内容。缺少静态来源时 build 明确报错。

额外发现的 AgentBenchFlow 只清点本地 task.md/SKILL.md；动态工具结果、任务资源、
生成对话无法从静态输入清单完整枚举，状态为 **partial**。自定义 synthetic task
seed/参数也需要重新建清单。本次没有将“未检查”记为“零污染”。

筛查包含：

1. NFKC、casefold、空白归一后的 exact document/passage 哈希及 query provenance。
2. **Qwen3.8 tokenizer** 的全部连续 50-token windows；滚动哈希定位后再比较实际
   50 个 token，排除哈希碰撞。另检查至少 12 个词的完整 evidence sentences。
3. 归一化 5-word shingles 的 MinHash/LSH 候选，64 permutations、16 bands，记录
   exact shingle Jaccard ≥0.65 的候选和评测来源。候选检测不是完整语义相似度证明。

为避免对大评测库保存全部 n-grams，先索引 bounded 训练候选，再流式扫描评测库，
不做两两全量比较。筛查结果缓存键绑定 source records、revision、tokenizer 和评测
manifest。扩大候选规模后需要重新扫描；大规模数据可按候选批次复用同一流程。

生成数据在 `outputs/`，原始获取缓存/SQLite 在 `.cache/`，均由项目 .gitignore 排除。
输出目录必须全新，防止覆盖旧实验。

## 运行命令

使用已有隔离环境 `.cache/cacherepair-qwen-env`（Torch 2.13.0+cu129、Transformers
5.8.0），没有全局升级/降级。它与官方 Torch 2.6 环境不同，属于 Qwen3.8 兼容适配，
不保证与原论文逐位数值一致。

```bash
# 只清点下载规模、revision、access/license
.cache/cacherepair-qwen-env/bin/python -m methods.CacheRepair.qwen38.prepare inventory \
  --output .cache/cacherepair-qwen38-source-inventory-new.json

# 建立所有本地评测输入的可复用内容清单
.cache/cacherepair-qwen-env/bin/python -m methods.CacheRepair.qwen38.prepare evaluation-inventory \
  --data-root data --skills-root /data/lyh/skillsbench \
  --output .cache/cacherepair-qwen38-evaluation-new

# 先去重，再按源文档关联组 split，最后编译 Qwen token chunks
.cache/cacherepair-qwen-env/bin/python -m methods.CacheRepair.qwen38.prepare build \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --evaluation-inventory .cache/cacherepair-qwen38-evaluation-new \
  --output outputs/cacherepair-qwen38/stage-a-data-new \
  --records-per-source 16 --passages-per-query 6 \
  --train-size 32 --validation-size 8 --max-tokens 256 --max-chunk-tokens 128 \
  --max-chunks 4 --seed 7

# 显式限定 statistics 为 8 个固定训练例子；省略参数则使用完整训练集
.cache/cacherepair-qwen-env/bin/python -m methods.CacheRepair.qwen38.train \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --dataset outputs/cacherepair-qwen38/stage-a-data-new \
  --output outputs/cacherepair-qwen38/stage-a-training-new \
  --gpus 1,2 --stats-examples 8 --cache-pairs
```

Stage A 配置见 `configs/Qwen3.8-27B-StageA.json`。目标实际模块与 native hybrid
cache 先经过几何检查，期望 `[B,T,16,4,2,256]` 仅作为断言。pair 的原生目标前向
成本若超过 120 秒则停止并保留 failure report，不改变 loss、验证要求或模型。

可选 `--cache-pairs` 只在**新实验目录**中保存冻结目标生成的 KV pairs；之后按需
读取，用于本次 statistics/train/diagnostics 重用，不在 GPU 保存完整数据集。
不跨实验目录复用，以免把其他目标权重生成的 KV 当作当前监督。

输出包括 `statistics.safetensors`、官方 `loss.jsonl`、`checkpoint-N/`、
`training-config.json` 和 `diagnostics.json`。诊断记录 validation 与零 residual
baseline 的 normalized MSE、原始 residual relative L2、预测量级、每层 K/V error、
训练吞吐、内存、时间、参数量和 checkpoint 大小。降低训练 loss 不代表回答质量提升。

### Stage B

扩大到每来源前 96 条 query、每 query 前 12 个 retrieval passages。对扩大的候选集
重新扫描同一个评测内容清单，再按源文档关联组划分 256/32 个例子。每例保留
2..8 chunks，单块最多 256、总计最多 1,024 tokens；实际长度可小于预算。
来源仍是有顺序偏差的 prefix sample，没有以验证误差选择来源、seed 或样本。

Stage B 是独立初始化的 pilot，不读取 Stage A 权重或 optimizer state。使用相同的
initialization seed，并从 Stage B 的固定 64 个训练例子重新估计 RMS；这两轮的
数据和 RMS 不同，不能把两个报告的绝对 normalized MSE 当成同一条学习曲线。
训练入口启动前检查配置对应的 split 数量、chunk 数和长度上限；有效 batch 固定为
官方循环的 4，不接受与之矛盾的配置。

```bash
.cache/cacherepair-qwen-env/bin/python -m methods.CacheRepair.qwen38.prepare build \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --evaluation-inventory .cache/cacherepair-qwen38-evaluation-v1 \
  --output outputs/cacherepair-qwen38/stage-b-data-new \
  --records-per-source 96 --passages-per-query 12 --workers 18 \
  --train-size 256 --validation-size 32 --max-tokens 1024 --max-chunk-tokens 256 \
  --max-chunks 8 --seed 7

.cache/cacherepair-qwen-env/bin/python -m methods.CacheRepair.qwen38.train \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --dataset outputs/cacherepair-qwen38/stage-b-data-new \
  --output outputs/cacherepair-qwen38/stage-b-training-new \
  --config methods/CacheRepair/qwen38/configs/Qwen3.8-27B-StageB.json \
  --gpus 1,2 --cache-pairs
```

Stage B 配置显式指定 statistics_examples=64；`--stats-examples` 可覆盖该值。
配置未指定且 CLI 未覆盖时，仍遵循官方默认的完整训练集统计。每层 K/V 的验证
误差与零 residual 基线使用相同 token 权重，总体 MSE 则遵循官方按 example 平均。
额外的 `experiment-config.json` 和 diagnostics 中的数据 SHA256 记录本次实际配置与
输入文件，不只记录临时上游配置。

```bash
.cache/cacherepair-qwen-env/bin/python -m pytest -q \
  tests/methods/test_cacherepair_qwen38.py \
  tests/methods/test_cacherepair_qwen38_data.py \
  tests/backends/test_cacherepair_qwen38_native.py \
  tests/backends/test_cacherepair_qwen38_training.py
```

## 按文档构建混合数据与固定任务验证

训练和验证共用 `document_data.py` 的 `task-documents-prefix-v1` 协议：

- 多文档样本按原顺序每文档一块，首块加入对应 Task 的 `prefixPrompt`。
  文档之间不合并、不按固定 token 数重新切片；保留最多前 32 个源文档。
- 单文档样本优先按自然段落、对话轮次或句子分成最多 4 块；缺少这些边界时
  按词间空白切分。无法形成两块的训练来源会明确排除并记录。
- 总长度上限为 16,384 tokens（包含前缀）。预算不足时公平分配给各文档，截断
  各自尾部并保留首块指令；不会让一篇超长文档挤掉后续文档。截断的 KB 文档
  保留末尾分隔符。最大单块可以超过 512，应读取输出 manifest 的实际最大值。
- RULER 保留生成输入自带的指令，不额外添加无关 QA 前缀；自然边界分块不要求
  运行 shuffle 或完整 KVBench 生成流程。CoRAG 保留样本的旧 chunks 逐 token 不变。

`PrepareCacheRepairTaskValidation.py --reformat-existing PREVIOUS_DATA` 使用原验证
manifest 的 500 个源记录及顺序，仅重新构造输入，不按模型分数重选数据：

```bash
python scripts/PrepareCacheRepairTaskValidation.py \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --audit outputs/cacherepair-qwen38/task-data-audit-20261028 \
  --reformat-existing outputs/cacherepair-qwen38/task-validation-v2-20261028 \
  --output outputs/cacherepair-qwen38/task-validation-documents-new
```

`MergeCacheRepairTrainingPool.py` 的原文 SQLite 池保留完整文档及有序上下文引用。
`CompileCacheRepairTrainingPool.py` 默认使用上述文档协议，并要求验证 manifest 的
协议相同。旧 raw512 方式只保留为显式 `--chunk-policy raw512` 历史选项。
完整输入去重、held-out 正文保护及连续 50-token 精确匹配筛查仍执行；公共任务
前缀从 held-out 索引中移除，避免将共享指令当成正文污染。排除记录另存
`compilation-exclusions.jsonl`。这不是穷尽语义改写的污染保证。

RULER 训练样本由 `PrepareCacheRepairRulerTrain.py` 使用固定 revision 的官方代码
独立生成，完整输入及目标标识避开当前测试/验证源和历史冻结测试；NIAH 另检查
独立背景长文本 anchors。来源 ID 包含完整输入哈希，避免原生 NIAH index 重复。
固定任务模板、词表和噪声允许共享，因此不套用真实数据的通用 span 排除规则。

## 定时验证和每个 epoch 的质量评测

`RunCacheRepairTrainingFast.py --evaluation-plan PATH` 将计划传给训练进程。
准备好的新微调作业已接上此参数；保存脚本不会启动后台调度器或训练。

- 首次更新前，沿用初始 checkpoint 的 RMS，对固定的 500 条验证数据测基线。
- 每次验证完成后计时 3,600 秒，到下一个 optimizer update 边界运行完整 500 条
  MSE。复用常驻的三组 target，保留 Python/CUDA RNG 与 model mode，再恢复同一
  epoch 的原始样本顺序和下一个位置；无须重载 27B。验证前保存 optimizer、RNG
  和数据游标，验证失败也能恢复到该更新。
- 每个 epoch 自带的 500 条验证抵扣相邻小时验证，避免重复。部署 checkpoint
  的 MSE 和重载一致性通过后，写入持久化质量评测请求，干净退出 trainer；
  supervisor 释放 target 后调用 `EvaluateCacheRepairEpoch.py` 跑相同冻结输入的
  10 tasks × 64 例，只生成当前 Hybrid checkpoint，复用有哈希保护的 Full/TRR。
  同时检查每例输入、缓存协议、checkpoint 路径、30 组计分与预测一致性，保存
  quality/TTFT 表格。完成后用 `--resume auto` 恢复完整训练状态。
- 质量评测失败不会自动跳过或继续训练，保留完整状态供重试；外部停止不触发
  自动恢复。已完成的 epoch 评测有持久化标记，重启不重复生成。

`training/scheduled-validation/` 保存基线、小时及 epoch MSE；
`training/validation-schedule.json` 保存下一次时间；
`quality-evaluations/step-*/completed.json` 指向该 epoch 的质量结果。
这是按验证结束重新计时的串行评测，不会同时启动多个小时验证。

## vLLM 教师的实验性 KV 核查

`scripts/AuditCacheRepairVllm.py` 在独立进程比较 native 与 vLLM TP=2 的同一批
token IDs，生产训练仍使用 native。vLLM 禁用 prefix cache、chunked prefill、
异步调度和 CUDA graph，使用 BF16 weights/KV、FP32 GDN state，不量化。
worker extension 捕获实际写缓存前的 post-K-norm/post-RoPE K/V，校验请求正文
token IDs、16 个 attention 层与 TP head 顺序，再使用同一文档位置、rerotation
和 canonical 坐标协议比较 stale/joint/delta，附每层 K/V 误差。

数值相近不会被报告为相同。`comparison.json` 只有逐位相同时才允许迁移；
不满足时保持 native，不改 checkpoint/RMS，也不自动启动训练。捕获实现使用
vLLM 内部接口，仅是版本绑定的核查工具，不是已通过验证的生产后端。
CPU canonicalization 与 native GPU canonicalization 可能有微小舍入差异；报告
单列不受 RoPE 影响的 V，使实际前向差异可以独立确认。测速包含 KV 导出和
canonicalization，是少量单次实验，不能用来承诺整轮训练或六卡利用率。

## 从 epoch 3 开始新的微调作业

`RunCacheRepairTrainingFast.py --init-from CHECKPOINT` 继承权重和原 RMS，使用新的
优化器、shuffle 和 LR 日程，不重跑 RMS。新数据必须用新作业目录；不能当作旧
50k 作业的 exact resume。已有新作业重启时自动使用其 `--resume auto`，保留优化器、
RNG 与下一批游标；可改为 4/2 张卡，目标仍按两卡一组运行。

训练配置可用 `training.learning_rate_scale` 显式缩放官方 LR 公式，默认 1 保持原值。
例如 1/3 对应 peak 1e-4、minimum 1e-5；cosine 周期使用该新作业的实际样本数与
`schedule_epochs`，warmup 保持周期更新数的 2%。配置和启动计划记录具体步数。

16k 输入可启用 `--memory-efficient-repairer`：FP32、精确 efficient SDPA，编码器和
残差头按 256-token tiles 工作，使用非重入 activation checkpointing 重算中间量。
参数名、checkpoint 格式和 MSE 目标不变；保存激活与浮点归约顺序可不同。
`long_training.py` 的 CPU 对照覆盖预测、全部参数梯度、验证权重及 exact resume。
验证诊断也分 tile 累积，避免一次保留多个完整 16k KV 预测张量。

实际长输入的 GPU 显存、六卡运行及减少卡数后的恢复，仍需在正式启动前进行真实
目标预检。生产 supervisor 要求预检报告，不应复制旧 4k 报告冒充新 16k 验证。
准备脚本只生成数据、配置和待手动调用的命令，不启动训练。监视默认每小时一次，
GPU 等待队列每 30 分钟检查一次；不设置自动暂停期限。

## MSE 对比口径

`joint - stale` 会随缓存边界与任务前缀改变。raw512、无前缀结果不能直接作为旧
文档块协议下的泛化分数。旧 500 条继续作为原协议回归集；新固定 500 条用一致的
文档/前缀协议验证。2026-10-28 调查详见
`outputs/cacherepair-qwen38/mse-investigation-20261028/REPORT.md`。

Normalized MSE 按 example 平均，原始 KV 空间的改善率/cosine 按元素汇总时，
坐标与权重均不同，需分别解读。评估沿用所评 checkpoint 的 RMS，不为改善数值
另行计算验证 RMS。验证损失不等同生成质量。
