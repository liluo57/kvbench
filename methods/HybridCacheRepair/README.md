# HybridCacheRepair

[十任务评测报告](EVALUATION_REPORT.md)：Full Prefill、TRR、Stage B Hybrid 各 640 例，
包含逐任务质量、完整在线 TTFT 和实现边界。

Qwen3.8 的混合缓存 Method：GDN 使用现有 Hypic TRR，全注意力 KV 使用训练后的
`qwen38` CacheRepair。公共接口：

```python
from methods import HybridCacheRepair

method = HybridCacheRepair(gpuNums=2, memFractionStatic=0.8)
```

`config.yaml` 的 `ModelPath` 必须指向训练时的 Qwen3.8 目标；
`HybridCacheRepair.RepairerPath` 指向本地 checkpoint。当前默认配置记录 Stage B
`outputs/cacherepair-qwen38/stage-b-training-v1/checkpoint-64`。

`HybridCacheRepair.ResidualAlpha`（默认 `1.0`），或构造参数 `residualAlpha`，
控制 `stale + alpha * (delta_norm * sigma_delta)`。缩放在 canonical KV 的 FP32
空间进行，再按原路径 materialize 并写回 BF16 cache；不缩放 stale 或 GDN state。
`alpha=0` 是相同 Hybrid 路径的 naive KV 对照，GDN 仍为 Hypic TRR。
评测脚本也支持 `--residual-alpha`；每例元数据和 rank-0 repair 日志记录实际 alpha。
构造函数只保存配置，模型与 CUDA 在 Engine 的 Method worker 内初始化。
使用主项目的 Hypic 环境，`./setup.py --hypic` 构建的 ops 继续复用。

## 状态与 KV 路径

1. `Prepare` 执行原 Hypic system/chunk/tail 暖身，保留原生 TRR 的 S/T、conv tails
   和公共 KV。另外对每个真实 chunk 做零状态、普通 causal prefill，将独立 KV
   保存到本 Method 的 CPU store。此路径与 repairer 训练输入对齐；Hypic 的暖身
   system/tail 不进入 repairer 的文档输入。
2. `Run` 复用现有 token-ID 匹配与 segment 边界，先执行原生 TRR donor，并通过
   既有状态捕获接口取得 query 前的 GDN state 和卷积历史。
3. 取已准备的独立 KV；fresh prefix spans 在线独立 prefill。按真实 RUN 顺序构建
   stale attention KV、token IDs、chunk IDs，排除最后的 query segment。
4. 使用目标运行时的真实 partial RoPE 表，在正确的 local/global 位置下重定位、
   canonicalize，执行 FP32 repairer，再 materialize 到请求私有 KV slots。
5. 恢复捕获的 TRR prefix state 和 conv history，通过现有 TRR 后缀路径重新执行
   query，然后原生 decode。没有 prepared hit 时使用普通 full prefill。

此实现复用原生 TRR donor 的缓存来源：其 GDN state 已包含 TRR attention 的影响。
修正 prefix KV 后重新计算 query，不重算整段 prefix 的 GDN state。因此对照比较的是
整个 hybrid 缓存策略，而非隔离所有交互后的单一 attention 模块效应。

公共 prepared KV 与独立 CPU store 保持不变。修正后的在线状态不发布成新的独立
chunk；沿用已有 request-private / LinearKV transient cleanup 释放所有临时资源。
`Reset` 同时清空 PICache 和独立 KV store。`retainOutput=True` 沿用 Hypic 的显式拒绝。

## 精度、TP 与显存

检查 checkpoint 格式、坐标、config/index 文件哈希和下载 ancestry，并核对运行时
真实层映射、KV head 数、维度、embedding 和 RoPE。哈希声明不等于全部权重的内容
哈希；继承训练产物相同的 provenance 边界。

目标 BF16、repairer FP32。TP 下按全局 head 顺序 all-gather KV，并从 frozen
vocabulary shards 读取完整 token embedding；不复制完整 target embedding。
支持 KV heads 均匀分片的普通 TP，拒绝 PP/DP/投机解码及量化目标。

输入 encoder 与输出 residual head 按 `kvTileTokens` 分块；repair blocks 对完整
前缀做一次 chunk-causal attention。推理包装复用原参数和官方 attention 的相同
QKV/RoPE/projection 公式，传入对齐的 additive mask，避免每个 block 复制 mask。
不截断、不做窗口化、不改变模型参数或 loss。CUDA 路径要求 FP32 efficient SDPA，
失败时明确报错。官方 vendored 源码保持原样。

单元测试验证非零权重下流式与原模型的数值一致性、native partial RoPE、TP
embedding、private-only KV 写回、prepared store 不变及 token-ID/retention 契约。
正式评测前还执行双卡真实目标烟测和 32K kernel/显存检查。

## 评测与计时

```bash
python scripts/EvaluateHybridCacheRepair.py \
  --config outputs/hybrid-cacherepair-10task-20261022/eval.yaml \
  --frozen outputs/kvbench-evaluation/qwen38-test-v1/frozen_tasks.pkl \
  --output outputs/hybrid-cacherepair-10task-20261022 --samples 64
```

`--frozen` 是可信本地输入快照；新评测使用 `outputs/kvbench-evaluation/qwen38-test-v1/frozen_tasks.pkl`。
`--samples 64` 从统一后的 10-task 新样本快照中按固定顺序取每个任务前 64 例。脚本逐例检查 case IDs、
完整输入 token hashes 和 generation budgets，使用相同输入比较 Full Prefill、Hypic TRR 和 Stage B Hybrid。
旧的 `hypic-ablation-64-20261014` 快照保留作历史记录，不应与新快照的结果当作同一测试集比较。
三组使用相同 BF16、普通 TP、eager/TRITON GDN、FlashInfer attention、greedy sampling。
不改写任务提示词；质量指标由原有 Task 评分实现计算。

TTFT 从 `Run` 开始，覆盖匹配、TRR donor、CPU/GPU 与 TP 传输、fresh-span prefill、
repairer、坐标转换、KV 写回、query replay 和首 token。初始化及 `Prepare` 在 TTFT
之外。当前 donor/query replay 实现用于验证效果；其完整在线开销保留在报告中。
Stage B 训练长度上限 1,024，而评测可能到 32K，应按实测质量解释长度外推表现。

完成后生成逐任务质量、配对差值、bootstrap 区间和延迟报告：

```bash
python scripts/SummarizeHybridCacheRepair.py outputs/hybrid-cacherepair-10task-20261022
```

脚本要求 Engine 成功结束、三组 case IDs 与输入哈希逐例一致、数量完整且数值有效。
运行中可加 `--allow-partial`，仅产生标记为未完成的 `progress.*`；完整结果为
`comparison.json`、`comparison.csv`、`comparison.md`。不同质量指标不混合成整体平均分。
