# Qwen3.8-27B 全量训练与恢复

本轮按用户选择**从头初始化**，不读取 Stage B 权重。配置
[`Qwen3.8-27B-Full.json`](configs/Qwen3.8-27B-Full.json)：50,000 train、500 validation，
6 epochs、batch=4，共 75,000 optimizer updates；按 8 epochs / 100,000 updates 的
官方学习率周期执行（2,000 warmup，peak 3e-4，min 3e-5）。AdamW、归一化 residual
MSE、梯度累积、clipping 和模型结构保持不变。目标 BF16，repairer/KV 及最终 RMS buffers
为 FP32，RMS 平方和累积为 FP64。

当前六卡续训作业为 `outputs/cacherepair-qwen38/full-training-repaired-v5`，启动/恢复命令见
[`PIPELINE_TRAINING.md`](PIPELINE_TRAINING.md)。RMS 已于 2026-10-23 18:17 完成全部 50k；
修复版本从原作业 step 838 迁移全部 optimizer/RNG/游标后继续训练，不重跑统计、不重置 LR。
下文旧单副本监督器示例作为早期作业说明保留。

本页描述配置与作业管理，**不表示全量训练已经完成**。实际阶段查看作业目录的
`job-status.json`，完成结果以 `training/status.json` 和 diagnostics 为准。

## 数据

`full_data.py` 固定此前检查过的 CoRAG query/corpus revisions，下载约 11 GB Parquet，
在每个来源内按 seed=7 无放回随机抽样，共 150,000 个候选 queries，读取前 20 个
retrieval IDs。下载复用 HF cache，已提交 corpus shards 记录在 SQLite 中，重试不必
重新抽样或重新解析已提交 shards。不是 pilot 的 first-N prefix sample。

最终每个 split 的 short/medium/long 比例严格为 10%/10%/80%，区间为
1..1024、1025..2048、2049..4096 **实际格式化后的 Qwen3.8 tokens**。每个区间内
ELI5/FEVER/NQ/WoW/T-REx/Structured Zeroshot 比例为 20/20/20/20/10/10。
保留 2..20 个 surviving passages 的检索顺序；单 passage 最多 512 tokens，第一块
另容纳真实 MuSiQue instruction。整块文本一次编码、无额外 chat template，保留尾部分隔。

这沿用[论文附录 J](https://arxiv.org/html/2609.35139v1#A10)的规模、来源比例和区间，
但使用新 manifest、Qwen3.8 tokenizer 和 KVBench 格式，不能声称精确复现官方训练集。
论文的长度统计不含 separators/instructions，且采用 Qwen2.5 tokenizer，两者有区别。
若清洁候选不足以满足比例，明确失败，不会重复填充或悄悄缩小训练数据。

筛查合并此前完整的本地静态评测清单和上轮实际 640 个评测输入曝光清单。
规则仍是 normalized exact text、逐项验证的 50-Qwen-token spans、evidence sentences、
query provenance 和 MinHash/LSH 候选；未审查近重复隔离。50-token postings 改为排序的
NumPy 数组，避免全量规模下的 Python set 开销，匹配后仍检查实际 tokens。
AgentBenchFlow 的动态内容仍是 partial，不能宣称完整语义去污染。

先保留独立 validation，随后从训练候选中排除其 Wikipedia 文档、内容、共享
50-token spans、LSH 近重复和问题文本，再构造训练 chunks。避免所有 retrieval queries
通过热门文档连成巨型 component 后无法划分。最终再次断言源文档无交叉。

全量统计默认遍历所有 50,000 个训练例子。每 epoch 验证全部 500 例；最终额外诊断
固定的 500 条训练例子，明确记录 row indices，不把子集指标冒充全训练集均值。
**不缓存完整监督 KV pairs**：FP32 stale+joint 每 token 262,144 bytes，约 1 亿 tokens
需要约 26 TB（十进制）。采用原生目标按需生成，代价由 GPU preflight 估计。

## 六卡加速作业

当前加速入口是 [`RunCacheRepairTrainingFast.py`](../../../scripts/RunCacheRepairTrainingFast.py)，
当前实现见 [`PIPELINE_TRAINING.md`](PIPELINE_TRAINING.md)，旧实现诊断见
[`FAST_TRAINING.md`](FAST_TRAINING.md)。它复用上述完整数据，重新初始化
repairer 和完整 50,000 条 RMS；三个两卡冻结目标副本并行提供监督，官方单进程 optimizer
保持 batch=4 和原始样本顺序。按用户最新指示，持续运行，等待用户要求时再停，
不按五小时自动暂停或缩卡；定期 checkpoint 保持启用。之后可选择 6/4/2 卡恢复。
旧 `full-training-v1` 的 Torch fallback 统计单独保留，不与 fused 后端统计混合。

## 作业与 GPU 等待

入口 [`scripts/RunCacheRepairTraining.py`](../../../scripts/RunCacheRepairTraining.py)：

```bash
.cache/cacherepair-qwen-env/bin/python -u scripts/RunCacheRepairTraining.py \
  --job outputs/cacherepair-qwen38/full-training-v1 \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --cache /data1/lyh/kvbench-cache/cacherepair-qwen38-full-v1 \
  --config methods/CacheRepair/qwen38/configs/Qwen3.8-27B-Full.json \
  --evaluation-inventory .cache/cacherepair-qwen38-evaluation-v1 \
  --evaluation-inventory outputs/hybrid-cacherepair-10task-20261022/evaluation-exposure-inventory
```

顺序：CPU 准备与去重 → 两张 GPU 空闲 → 实际 native forward/backward preflight →
完整 RMS → 六 epoch 训练。数据规模和代码/config SHA 均校验后才训练。
排队每 **1800 秒**检查一次，不允许配置成更高频。空闲要求：无 compute process，
显存使用 ≤1024 MiB，利用率 ≤5%，总显存 ≥44000 MiB。不会停止其他 GPU 进程。
可用 `--allowed-gpus 1,2,3,4` 限定物理卡；默认任意两张满足条件的卡。
使用文件锁防止本工具的重复作业互相抢卡，其他程序仍可能并发申请 GPU。

preflight 测试每个长度区间中最长和 chunk 数最多的真实训练样本，执行真实目标、
repairer forward/backward、optimizer step，记录内存/时间及保守耗时估计。
预检或子进程失败时停下并记录错误，不改变目标、数据规模或精度继续运行。

`job-status.json` / `events.jsonl` 提供阶段与 GPU 检查时间；`preparing_data.log`、
`preflight.log`、`training.log` 保存子进程日志。`supervisor.json` 记录 PID。
启动命令使用 detached process 时，关闭终端不终止作业。它不是系统开机服务。

## 两种继续训练方式

`train.py --resume CHECKPOINT`（或 `--resume auto`）完整恢复：

- FP32 repairer 与 RMS，AdamW moments/step，Python/CPU/CUDA RNG。
- 全局 update、学习率 horizon、下一个 epoch 和 batch offset。
- 原始每 epoch shuffle；中间 batch 不重复计入已提交训练。
- 以数据/目标 provenance、模型结构、seed、训练周期、实现 SHA 和库版本校验一致性。
- loss 日志中超出 durable checkpoint 的行归档到 `replayed-loss.jsonl` 后重放。

每 250 次 update 和每个 epoch 末保存 `resume-checkpoints/step-XXXXXXXX/`。
模型与 `trainer.pt` 写完并 fsync 后原子提交；`latest.json` 指向已提交 checkpoint。
保留最近两份，当前恢复所引用的快照额外保护。每 epoch 的部署 checkpoint 单独保留。
SIGINT/SIGTERM 在当前 optimizer update 完成后额外保存并退出；SIGKILL/机器崩溃
恢复最近 durable snapshot，可能重做最多 249 次尚未提交的 updates。
完整 RMS 使用 FP64 累积，每 100 例保存 `statistics-progress.pt`；在统计阶段被中断
也可以用 `--resume auto` 继续。最终 RMS 公式仍执行官方代码，不做分批 FP32 平均。

```bash
# 原目录恢复；普通中断后，重新运行上面的 supervisor 也会自动采用此选项
.cache/cacherepair-qwen-env/bin/python -m methods.CacheRepair.qwen38.train \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --dataset outputs/cacherepair-qwen38/full-training-v1/data \
  --output outputs/cacherepair-qwen38/full-training-v1/training \
  --config methods/CacheRepair/qwen38/configs/Qwen3.8-27B-Full.json \
  --gpus 1,2 --resume auto
```

`--init-from PATH` 仅加载**权重与已有 RMS**，使用新的 optimizer、shuffle/schedule 和新
输出目录；可以载入之前的 Stage B `checkpoint-64`。不将旧模型换成新 RMS 后直接训练，
因为那会改变输入与输出归一化意义。旧 Stage A/B checkpoint 没有 optimizer state，
支持 init-from，不能冒充 exact resume。两个参数互斥。

监督器收到 SIGINT/SIGTERM 会转发给训练进程并停止，不自动重启用户主动停止的训练。
需要暂停时，可向 `supervisor.json` 中的 PID 发 SIGTERM；确认 `interrupted` 和
`latest.json` 后，使用原监督器命令恢复。正常恢复不会覆盖旧实验目录。

## 代码与验证

官方 commit 仍是 `4eb96cd92df4be4f679ad8ac42a51442249683c2`，vendored
`train.py/model.py/cache.py` 逐字节不变。`resume.py` 对 main 的 AST 只增加恢复入口、
update 后保存、循环起点及日志追加；对 compute_stats 只增加累积状态恢复/保存。
各个匹配点有结构断言，上游变动时拒绝运行。loss、optimizer、LR、shuffle、batch 和
梯度累积保持官方语句。测试包括跨 epoch 的连续/恢复训练权重及 loss 逐项一致、
统计 FP64 恢复、旧权重初始化、数据比例/区间/隔离，以及不抢占繁忙 GPU 的选择条件。
