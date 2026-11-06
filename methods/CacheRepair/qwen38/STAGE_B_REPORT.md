# Qwen3.8-27B CacheRepair Stage B 报告

日期：2026-10-22。完成扩大数据的单 epoch pilot。使用真实本地 Qwen3.8-27B BF16 权重、原生 GDN + full-attention 前向，以及现有 54,017,664 参数 repairer。

后续进展：同日完成 HybridCacheRepair 在线集成及十任务评测，见
[在线评测报告](../../HybridCacheRepair/EVALUATION_REPORT.md)。本页保留 Stage B 训练阶段的记录。

Stage B 独立初始化，未读取 Stage A 权重或 optimizer state。初始化 seed=853119240、数据/训练 seed=7；RMS 来自固定的 64 条 Stage B 训练样本。目标与模型结构、官方 loss/optimizer/schedule 不变。

## 固定数据上的结果

| Split | 例数 | 零 residual normalized MSE | 训练后 MSE | 相对下降 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 256 | 0.946472 | 0.781323 | 17.45% |
| 验证 | 32 | 1.032684 | 0.858307 | 16.89% |

验证集原始 residual relative L2 error=0.892220；预测/真实 residual L2=0.403193；预测 residual/stale KV L2=0.178917。

总 MSE 按 example 平均；逐层表按 token 加权。零修正是当前零初始化输出头在同一数据和 RMS 下的基线。训练后训练集指标来自再次固定遍历，不能把不同随机 batch 的日志视为固定集学习曲线。

这些是训练来源 held-out 数据上的 KV residual 指标。未执行在线 Hypic Addition/TRR + Repairer 接入、回答质量或 TTFT 对照。Stage A/B 的数据与 RMS 不同，绝对 normalized MSE 不能直接跨轮比较。

## 数据与筛查

六来源各取前 96 条 query，共 576 条；按已发布检索顺序取前 12 个 passages，共 6818 个独立 passages。筛查 243,608 段本地评测内容、146,851,665 个 Qwen tokens。

确认污染并排除 637 个 passages；待审查并隔离 1 个。完整 passage exact duplicate=0，50-token span=326，evidence sentence=596；类型之间可能重叠，不能相加。

按 workload 的确认 passages：{"musique": 53, "hotpotqa": 76, "2wikimultihopqa": 55, "triviaqa": 492}；question provenance 命中 0 项，丢弃 query 共 4 条。MinHash/LSH 共 32 个匹配事件。详情见数据目录的 deduplication.json。

完整源内容先筛查，再按 query、Wikipedia 文档、内容、共享 spans 和近重复候选的关联组划分 train/validation，之后构造 chunks。训练/验证源文档互不交叉。所有可用本地静态评测内容均筛查；AgentBenchFlow 仅覆盖 task.md/SKILL.md，动态工具输出、任务资源和生成对话仍为 partial。Copy/FreshGap 限默认配置、seed=42、前 100 cases。

| 来源 | 训练 | 验证 | 排除 passages |
| --- | ---: | ---: | ---: |
| eli5 | 44 | 2 | 96 |
| fever | 44 | 5 | 213 |
| nq | 44 | 4 | 82 |
| wow | 40 | 7 | 223 |
| trex | 43 | 8 | 18 |
| structured_zeroshot | 41 | 6 | 5 |

train：共 110,789 tokens，长度 75..958；chunk count 分布 {'3': 45, '2': 34, '5': 39, '8': 35, '4': 35, '6': 41, '7': 27}。

validation：共 12,203 tokens，长度 131..925；chunk count 分布 {'3': 6, '2': 5, '4': 4, '6': 4, '7': 4, '8': 6, '5': 3}。

预算上限 1,024 tokens/例、256 tokens/chunk，2..8 chunks，保留真实 KB 格式并独立编码。该数据仍为固定 prefix sample，存在来源顺序偏差，不是官方 50,000 条语料的精确复现。

## 训练与资源

| 项目 | 实测 |
| --- | ---: |
| Effective batch / epoch / optimizer steps | 4 / 1 / 64 |
| 模型加载 | 39.06 s |
| RMS 估计 | 144.80 s，64 个例子 |
| 训练更新时间 | 440.85 s |
| 训练吞吐 | 0.581 examples/s；251.31 tokens/s |
| 统计、训练、验证、最终训练诊断合计 | 665.72 s |
| Checkpoint | 206.317 MiB，70 tensors |
| GPU 1 peak allocated / reserved | 27.237 / 28.590 GiB |
| GPU 2 peak allocated / reserved | 25.410 / 25.939 GiB |

按需生成 288 个唯一 supervision pairs，累计 122,992 tokens、594.21 s；run-local disk cache 命中 321 次。没有在 GPU 常驻完整监督数据集。

目标 attention 使用 eager，GDN 使用当前 Transformers 的 PyTorch fallback，未安装 FLA/causal-conv1d。训练计算和权重放在 GPU 1、2；运行时另外在 GPU 0、3、4、5 各观察到约 426 MiB 的 CUDA context 开销，未运行这些卡上的模型计算。gpu-process-snapshot.json 保存单次进程级观测；上表是 Torch allocator 的峰值，两种口径不同。后续可用 CUDA_VISIBLE_DEVICES 将可见设备限制在所用两卡。

全部 64 步 repairer 梯度有限且非零；目标全部冻结、无梯度、未进入 optimizer，抽样目标参数指纹前后一致。Checkpoint 重载输出逐元素一致，两个 RMS buffers 均为 FP32 [16,4,2,256]、有限且为正，不含目标 embedding。专项测试 34 passed。

## 每层 K/V 验证误差

| Native layer | K 零修正 | K 训练后 | V 零修正 | V 训练后 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1.078470 | 0.683048 | 1.211912 | 1.010110 |
| 7 | 1.103838 | 0.761721 | 1.207369 | 1.032171 |
| 11 | 1.173847 | 0.866339 | 1.239625 | 1.060206 |
| 15 | 1.158525 | 0.848090 | 1.216147 | 1.051962 |
| 19 | 1.124059 | 0.833874 | 1.218144 | 1.100259 |
| 23 | 1.143546 | 0.908250 | 1.173953 | 1.080266 |
| 27 | 1.104997 | 0.894306 | 1.115817 | 0.989630 |
| 31 | 1.044286 | 0.825553 | 1.068700 | 0.950062 |
| 35 | 1.010263 | 0.786170 | 1.059356 | 0.965790 |
| 39 | 1.006665 | 0.810469 | 1.020212 | 0.942725 |
| 43 | 1.001524 | 0.829152 | 0.996333 | 0.887642 |
| 47 | 0.992903 | 0.789783 | 1.011400 | 0.919887 |
| 51 | 0.993761 | 0.770363 | 1.016034 | 0.944137 |
| 55 | 0.991171 | 0.780231 | 1.031956 | 0.996236 |
| 59 | 0.992126 | 0.794878 | 1.043669 | 1.003694 |
| 63 | 0.983153 | 0.775798 | 1.005485 | 0.889625 |

## 版本、适配和产物

CR commit `4eb96cd92df4be4f679ad8ac42a51442249683c2`；官方 train.py/model.py/cache.py 与 checkout 逐字节相同。继续执行原函数代码，没有增加第二套训练循环。

本轮新增 Stage B 配置、启动前数据检查、实验配置/数据指纹记录、逐层 K/V 零修正基线。Corpus range-read worker 使用独立 spawn 进程，隔离 HF Hub 共享 HTTP 客户端在连接重试时的相互影响；数据来源、顺序和筛查算法未修改。

Query revision `d3f3f906826b1524e7db4301723cecad770be0a7`；corpus revision `bf95d9b4b19c68420395efad7498d0ef5052ab4b`。Tokenizer 内容 revision、目标下载 metadata 声明的逐文件 revision、config/index 哈希与实际环境版本均随产物记录。没有重新哈希全部目标权重；配置哈希不等于权重哈希。

- `outputs/cacherepair-qwen38/stage-b-data-v1/`：JSONL、provenance manifest、dataset.json、去重报告。
- `outputs/cacherepair-qwen38/stage-b-training-v1/checkpoint-64/`：repairer、RMS、目标兼容元数据和 target-provenance.json。
- `outputs/cacherepair-qwen38/stage-b-training-v1/`：loss.jsonl、statistics.safetensors、diagnostics.json、stage-b-summary.json、experiment-config.json。
- `.cache/cacherepair-qwen38-stage-b-data-v1.log`、`.cache/cacherepair-qwen38-stage-b-training-v1.log`：实际运行日志。

运行命令和数据策略见 [TRAINING.md](TRAINING.md)。Stage A 产物完整保留。后续优先完成实际 Hypic state composition + repaired attention KV 的在线集成，再做预先选定、已排除训练污染的 KVBench 质量/TTFT 对照。
