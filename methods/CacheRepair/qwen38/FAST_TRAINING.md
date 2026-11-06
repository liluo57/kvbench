# Qwen3.8 多卡监督与手动停止

本页保留 v2 串行双卡副本实现和诊断记录。当前流水线实现见
[`PIPELINE_TRAINING.md`](PIPELINE_TRAINING.md)，v2 的精确源码在旧作业的 `source-before-pipeline/`。

完整配置仍为 50,000 train、500 validation、6 epochs、batch=4、75,000 updates，
从头初始化。RMS 使用全部 50,000 条训练样本，不使用烟测的统计子集。
作业启动、RMS 统计、梯度训练是不同阶段：先完成 RMS，再执行第一步 optimizer update。

## 实现

- 原生 HF Qwen3.5 架构加载实际 Qwen3.8-27B 权重，BF16，SDPA attention，
  官方 FLA 0.5.2 / causal-conv1d 1.7.0 融合 GDN；缺少 fast kernels 时拒绝启动。
- `packed.py` 在一个目标 forward 中处理独立 chunks：attention 按 chunk 隔离，
  causal convolution 用 seq_idx 重置，GDN 用 cu_seqlens 重置；位置从各 chunk 的零开始。
  这些边界包装在 stale forward 后恢复，joint forward 仍使用原始完整上下文。
- `pair_pool.py` 在 GPU 0/1、2/3、4/5 分别加载两卡冻结目标。修复器和官方 optimizer
  留在第一个进程，监督预取有界，按原始 shuffle 顺序消费，不改变有效 batch。
- RMS 在每个副本本地执行官方 FP64 累积，以连续 20 例为块，按源顺序合并 token 加权的
  sums/count；跨进程只传小型统计。分块加法次序可能改变 FP64 最后几位，不声称逐位相同。
- 每 100 条统计或收到 SIGTERM 时保存统计游标与 FP64 sums/count；梯度训练每 100 updates
  保存修复器、RMS、AdamW、RNG 和下一 batch 游标。GPU 副本数量不进入恢复身份。

## 数值与性能验证

实际 27B 的 row 0（2,349 tokens）和最长 row 17,911（3,998 tokens）已执行原实现、
融合逐 chunk 和融合 packed 对照。单副本监督 pair 从 7.71/10.04 秒降到 2.58/4.77 秒，
为 2.99/2.11 倍；这是两个样本的目标生成微测，不是完整训练吞吐。

融合 GDN 与旧 Torch fallback 在深层 BF16 KV 上存在差异：joint KV relative L2 分别
4.73%/10.33%，末 token logits cosine 为 0.99853/0.99814，top-1 均相同。
packed 相对融合逐 chunk 的 stale KV relative L2 为 1.33%/4.14%，joint 相同，
logits cosine >0.99998，top-1 相同。新实验重新计算 RMS，不复用旧统计。
报告完整保留这些差异，不能声称数值逐位复现旧后端。

2026-10-23 实际完整 RMS 恢复后的 60 次采样（65.5 秒）中，GPU 0..5 平均利用率为
53.7%、39.2%、46.9%、53.6%、44.1%、50.9%，六卡均值约 48.1%。这是 RMS 计算，
不是梯度训练；首次加载、全数据校验及上游输入准备期间会出现 GPU 利用率为零。
原始记录位于作业的 `steady-utilization.jsonl`，不以瞬时峰值代表持续利用率。

FP32 CPU 参考测试单独检查 attention、convolution、GDN 的独立 chunk 边界，以及包装恢复。
最长真实样本已通过修复器 forward/backward 和 AdamW step，梯度有限、目标冻结。
GPU 峰值 reserved 约 35.7/34.9 GB。六卡 optimizer 烟测及两卡 checkpoint 恢复另有真实测试。
官方 vendored train/model/cache 文件保持不变，CPU 回归覆盖样本顺序、权重、loss、RNG 恢复。

## 启动与资源窗口

```bash
.cache/cacherepair-qwen-fast-env/bin/python -u scripts/RunCacheRepairTrainingFast.py \
  --job outputs/cacherepair-qwen38/full-training-fast-v2 \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --dataset outputs/cacherepair-qwen38/full-training-v1/data \
  --config methods/CacheRepair/qwen38/configs/Qwen3.8-27B-Full.json
```

监督器启动前要求真实目标验证和六卡训练/两卡恢复烟测均通过，保存代码、配置和数据 manifest
SHA。排队时每 30 分钟检查一次；不会停止其他 GPU 作业。运行时每分钟记录 GPU telemetry。

用户最新要求是持续运行，等用户要求时再停。因此默认无截止时间，
`resource-policy.json` 中 `deadline_unix=null`、`after_window=continue_until_requested`；
不会在 15:15 或五小时结束时自动暂停，也不会自动改变 GPU 数量。定期 checkpoint 仍然启用。
只有显式传入 `--run-hours` 才启用定时暂停，默认命令不传此参数。

手动停止后，使用原命令加 `--physical-gpus 0,1,2,3` 可恢复为四卡两副本；
改为 `--physical-gpus 0,1` 即两卡一副本，训练/RMS 状态不变。

`--attach-pid PID` 可让监督器接管仍在运行的本作业训练进程，不重新加载模型。
接管校验进程 owner、独立 process group、作业输出、resources-ready PID、CUDA_VISIBLE_DEVICES
以及 Linux starttime，避免将信号发送给复用的 PID。2026-10-23 取消定时停机时，
仅替换监督器；RMS 根进程 3993172 和两个子进程保持不动，接管记录在 `supervisor-handoffs.jsonl`。
显式停止后若退出超时 600 秒，仅终止本作业进程组，恢复最近 durable checkpoint。

## 稳态利用率瓶颈

`load_native_target` 将前 32 层放在一张卡、后 32 层放在另一张卡，逐输入串行执行。
这让 BF16 模型能容纳于两张 L40，但不是同时在两张卡上计算的 tensor parallel，
也没有交错输入的 pipeline。RMS 的每个 20 条块仍逐条 forward，native batch size 为 1。
三个副本彼此并行，副本内部的另一张卡仍需等待当前输入前半/后半层完成。

65.5 秒、60 次稳态采样中，三对卡的平均 utilization 之和为 93.0%、100.5%、95.0%，
各对两张卡利用率相关系数为 -0.724、-0.748、-0.728，支持双卡交替忙闲这一判断。
六卡 utilization 总和折合约 2.9 张持续运行 kernel 的卡；不等于 2.9 张卡的峰值算力。
GPU topology 为 PCIe，未显示 NVLink，跨卡激活/KV 转移还有额外成本。
各 target 进程 CPU 约占一核，但没有可靠 CUDA trace，不能将 CPU、FP64 归约和 PCIe 的
具体耗时百分比当成已测量结果。在线 py-spy attach 受当前 ptrace 权限限制。

进一步提升应优先让独立输入在两级 pipeline 中重叠，或实现适配 GDN 的 tensor parallel；
增加副本数量本身保留了上述每对卡内部的串行等待。

主要文件：`job-status.json`、`resource-policy.json`、`training-fast.log`、
`gpu-telemetry.jsonl`、`training-pair-events.jsonl`、`training/status.json`。
向 `fast-supervisor.json` 中的 PID 发 SIGTERM 会保存并停止整个作业；重新运行同一命令恢复。
进程独立于终端运行，但不是系统开机服务。
