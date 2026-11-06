# HybridCacheRepair：Qwen3.8-27B 十任务评测

2026-10-22。完成 Full Prefill、Hypic TRR、Stage B Hybrid 各 10 个 task × 64 例，共 1,920 条正式结果。Engine 状态为 completed，运行失败 0。

当前 Hybrid 有任务相关收益：相对 TRR，7 项主指标均值提高、2 项降低、1 项持平。MuSiQue、2Wiki、HotpotQA 的 F1 分别提高 10.74、7.81、10.03 分，但仍低于 Full Prefill。VT accuracy 从 TRR 的 100 降至 86.88。不能把当前 checkpoint 作为统一优于 TRR 的方案。

## 质量

各项按原 KVBench Task 评分，统一展示为 0–100。RULER 用 accuracy，QA 用 F1，摘要用 ROUGE-L；不同指标不混合成整体质量均值。accuracy 不等同于另存的 exact_match。Delta 和区间均为 Hybrid 减 TRR，按未舍入数值计算。

| Task | 指标 | Full | TRR | Hybrid | Delta | 配对 bootstrap 95% 区间 |
|---|---|---:|---:|---:|---:|---:|
| niah_shuffle | accuracy | 100.00 | 100.00 | 100.00 | +0.00 | [+0.00, +0.00] |
| cwe_shuffle | accuracy | 99.84 | 93.59 | 95.00 | +1.41 | [-2.19, +4.84] |
| vt_shuffle | accuracy | 85.62 | 100.00 | 86.88 | -13.12 | [-16.88, -9.69] |
| musique | f1 | 70.59 | 19.99 | 30.73 | +10.74 | [+1.21, +20.08] |
| 2wikimultihopqa | f1 | 69.11 | 39.52 | 47.33 | +7.81 | [-2.92, +19.48] |
| hotpotqa | f1 | 75.38 | 46.02 | 56.04 | +10.03 | [-2.92, +23.04] |
| triviaqa | f1 | 85.73 | 82.71 | 83.75 | +1.04 | [-2.08, +5.21] |
| samsum | rougeL | 28.23 | 27.33 | 28.61 | +1.28 | [+0.11, +2.58] |
| multinews | rougeL | 15.22 | 14.64 | 15.93 | +1.29 | [+0.51, +2.13] |
| govreport | rougeL | 20.66 | 19.15 | 18.91 | -0.24 | [-0.84, +0.39] |

区间使用固定 64 例的配对 percentile bootstrap，2,000 次重采样，seed=38。它只反映这批 case 的波动，不覆盖训练 seed、模型或固定 prefix 抽样偏差，也未作十任务多重比较校正。MultiNews、SAMSum、MuSiQue 的区间在 0 以上，VT 在 0 以下；其余点估计变化需要更多样本确认。

## 在线开销

640 例合并后的平均 TTFT：Full 1.295 s，TRR 0.211 s，Hybrid 0.689 s。Hybrid 为 TRR 的 3.27 倍，约为 Full 的 53.2%。短输入的 MuSiQue、SAMSum、2Wiki、HotpotQA 上，当前 Hybrid TTFT 高于 Full。此处是共享主机、Engine 多 worker 调度下的观测值。

| Task | Full TTFT / s | TRR TTFT / s | Hybrid TTFT / s |
|---|---:|---:|---:|
| niah_shuffle | 1.299 | 0.187 | 0.580 |
| cwe_shuffle | 1.306 | 0.209 | 0.607 |
| vt_shuffle | 1.338 | 0.190 | 0.589 |
| musique | 0.739 | 0.290 | 0.821 |
| 2wikimultihopqa | 0.358 | 0.236 | 0.668 |
| hotpotqa | 0.537 | 0.237 | 0.662 |
| triviaqa | 3.157 | 0.201 | 0.943 |
| samsum | 0.124 | 0.170 | 0.458 |
| multinews | 0.735 | 0.187 | 0.587 |
| govreport | 3.360 | 0.203 | 0.975 |

TTFT 包含 Run 内的匹配、TRR donor、GDN state 捕获、fresh-span 独立 prefill、CPU/GPU 与 TP 传输、KV 重定位、FP32 repairer、KV 写回、状态恢复、query replay 和首 token。初始化及 Prepare 在 TTFT 外。Hybrid 的 Prepare 另做独立 chunk prefill 并保存 CPU KV；其离线成本不能从 TTFT 推断。

640 例均实际调用 repairer，对应 640 条 rank-0 repair 事件；KV 读取/重定位/修复/写回阶段平均 0.181 s，此阶段不包含整个在线路径。完整 total time、输出长度、输出预算命中数和所有质量指标均保存在 comparison.json。GovReport、MultiNews、CWE、VT 中存在大量输出用满预算的例子，三组预算保持原样。

## 固定协议与模型

- 目标：本地 `/data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B`，原生 hybrid Qwen3.5 text backbone；64 层中 48 个 GDN、16 个 full-attention 层。
- GDN：原 Hypic TRR。KV：Stage B `checkpoint-64`，54,017,664 个 FP32 参数，RMS 随 checkpoint 加载；本轮没有训练或改选 checkpoint。
- Target BF16，TP=2，batch=1；三组相同 eager、TRITON GDN、FlashInfer attention、greedy sampling。Hybrid 使用 FP32 efficient SDPA，kvTileTokens=256；repair blocks 仍处理完整 prefix。
- MaxModelLen=32,768；真实输入为 64..24,857 tokens。Stage B 仅 256 个训练样本，训练预算至多 1,024 tokens，存在明显长度及任务分布外推。
- 使用历史可信本地 `outputs/hypic-ablation-64-20261014/frozen_tasks.pkl`。逐例核对 case IDs、prepare/run token hashes 与原 generation budgets，未重新渲染提示词。
- 实际 640 例输入与所选训练/验证来源的 1,398 个完整 passages 额外复核；扫描 4,843 个唯一评测内容单元、4,762,685 tokens，按既有规则确认命中 0、待审查近重复 0。筛查范围与阈值边界见 audit 产物及训练报告。

本轮比较的是整个 Hybrid 缓存策略：TRR 的 GDN prefix state 来自原暖身 KV donor；repairer 使用训练所要求的独立 causal chunk KV，修正 prefix KV 后重算 query。没有重算整个 prefix 的 GDN state。TRR 与 Hybrid 的 KV 来源也有差别；尚缺同路径零 residual 对照，无法把所有质量变化单独归因于学习到的 residual。

## 验证与产物

新增 Method 专项 13 passed；独立 HF 5.8 环境中的原生模型、训练通路和数据回归 21 passed。完成双卡原生目标烟测、真实最长样本以及流式/完整模型的非零权重数值对照。

32,753-token kernel 检查输出全部有限，peak allocated 5.236 GiB；该独立测试使用小 embedding stub，不代表完整目标服务总显存。Meta + assign 加载检查在默认 BF16 下仍保持 70 个 checkpoint tensors 全为 FP32、数值逐元素一致。

完成核对：30 个 task/method 结果组各 64 例；Engine 分数/TTFT 均值与逐例记录一致；固定输入文件、已记录的运行源码和 checkpoint 哈希未变；官方 CR train.py/model.py/cache.py 与 commit 4eb96cd92df4be4f679ad8ac42a51442249683c2 checkout 逐字节相同。目标兼容检查覆盖 config/index 哈希及下载 ancestry，未声称重新哈希全部目标权重。

退出时有原生 multiprocessing resource_tracker 的 semaphore/shared-memory 清理 warning。四个 Engine workers 均正常关闭，GPU release 事件确认相关 GPU 上没有本实验的 compute PID，Engine 无运行或释放失败。

- [Method 与复现命令](README.md)
- [机器可读汇总](../../outputs/hybrid-cacherepair-10task-20261022/comparison.json)、[CSV](../../outputs/hybrid-cacherepair-10task-20261022/comparison.csv)
- [Engine 完整报告](../../outputs/hybrid-cacherepair-10task-20261022/engine_report.json)、[逐例预测](../../outputs/hybrid-cacherepair-10task-20261022/predictions/)
- [完成核对](../../outputs/hybrid-cacherepair-10task-20261022/completion-audit.json)、[源码/checkpoint 指纹](../../outputs/hybrid-cacherepair-10task-20261022/source-provenance.json)
- [运行配置](../../outputs/hybrid-cacherepair-10task-20261022/eval.yaml)、[输入清单](../../outputs/hybrid-cacherepair-10task-20261022/selection.json)、[环境](../../outputs/hybrid-cacherepair-10task-20261022/runtime-environment.json)
- [额外训练/评测内容复核](../../outputs/hybrid-cacherepair-10task-20261022/training-exposure-audit.json)、[Stage B 训练报告](../CacheRepair/qwen38/STAGE_B_REPORT.md)
