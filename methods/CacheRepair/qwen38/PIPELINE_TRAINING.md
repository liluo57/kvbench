# Qwen3.8 原生双卡流水线

当前作业目录为 `outputs/cacherepair-qwen38/full-training-repaired-v5`。完整训练仍为 50,000 条
RMS、50,000 train、500 validation、6 epochs、batch=4，实验最初从头初始化修复器。
RMS 于 2026-10-23 18:17:32 完成；v4 在第 838 次 optimizer 更新后受控保存，v5 通过
`scripts/MigrateCacheRepairTraining.py` 迁移全部数值状态后继续，未重跑 RMS、未重置 optimizer 或 LR。
迁移凭据位于当前作业的 `training/migration.json`，旧版本及原始 checkpoint 保留在 v4。
RMS 只保存 FP64 累积和归一化尺度，完整监督 KV 按需生成，用完释放。
默认持续运行，等待用户要求时停止；不按五小时自动暂停或减卡。

## 调度与状态隔离

三个副本分别使用两张卡，前 32 层和后 32 层各自受阶段锁保护，允许不同输入在两半模型
上重叠。目标仍是原始 HF Qwen3.8-27B、BF16、SDPA、官方 fused GDN。
每个输入使用独立 CUDA streams、native cache 和 GDN chunk 边界；边界放在线程局部状态中，
不会将 stale 的卷积或递归重置规则带入 joint forward。
跨流发布前等待生产流完成，并记录消费流，保护 autograd 保留的目标输入。

RMS 各副本有三个在途输入，连续处理 100 例后合并小型 FP64 sums/count。
v4 将主副本的 RMS 放入独立线程，并为每个副本预排两个统计块，避免主线程等待远端时
本地主副本停止计算。使用 `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`，减少原生分配器
反复回收、重试造成的停顿；没有启用曾在恢复烟测中触发 CUDA 错误的 expandable segments。
块按数据源顺序合并，逐 token 加权。块长从 20 改为 100 可能改变 FP64 最后几位的加法顺序；
不声称跨分块方案的完整 50,000 条统计逐位相同。
检查点间隔相对于已恢复的游标计算，确保从 3,380 这样的游标继续时仍每 100 例保存。

梯度训练保持官方单进程 optimizer、原始 shuffle 和 batch=4。v5 的修复器副本及远端副本均保留
三个在途目标输入。真实六卡工程检查中，主副本前三个输入均为 3,998 tokens，完成 backward
后主卡峰值为 36,268 MiB；step 1 的全部权重/RMS 与原两路主副本调度逐位一致。
证据位于 v5 的 `root-depth3-proof.json`。跨进程请求携带后续输入，
目标流水线可以在 IPC 回调之间继续计算，结果仍按原始顺序消费。
RMS 的利用率测量不能替代后续梯度阶段的利用率测量。

## 旧统计导入与验证

v3 已保存 3,380 条 RMS（6,751,947 tokens），其中包括 v2 导入的 2,680 条。
v4 导入只复制 FP64 sums、token count 和游标，不导入旧权重或 optimizer。
`--statistics-from` 要求数据、目标、seed、周期、库版本和数值后端一致，并校验原统计的行身份。
唯一允许的执行差异是流水线深度及实现版本；还必须提供真实目标数值验证证据和当前代码 SHA。
`statistics-import.json` 记录来源和范围，v2、v3 的统计及精确源码保留在原目录。

真实 27B 检查涵盖 9 个训练样本（含最长 3,998 tokens），比较完整 stale/joint KV 和单块
FP64 RMS 累积的逐位相等。六卡烟测使用明确标注的工程 fixture，包含多个不同 token 的
最长输入，执行真实 backward、AdamW 和 checkpoint；四卡恢复沿用同一训练身份。
工程 fixture 不作为正式训练数据。

v4 在 cudaMallocAsync 下的真实 27B 验证中，9 例流水线耗时 15.15 秒，相对串行约快 1.60 倍，
完整 stale/joint KV 及单块 FP64 平方和逐位一致。六卡梯度 step 1、四卡精确恢复 step 2 均已通过。
这些为工程验证，不表示正式 RMS 已完成或正式梯度训练已开始。

v3 的六卡稳态采样为 180 次、约 205 秒，各卡平均利用率分别为
88.03%、83.11%、87.03%、91.82%、88.01%、86.23%，尚未达到每卡 90% 的目标。
v4 的首段稳态采样为 180 次、约 179 秒，各卡平均利用率分别为
88.49%、88.56%、83.67%、90.52%、85.82%、86.14%，六卡均值 87.20%，仍未达到每卡 90%。
原始采样及摘要写入作业目录下 `steady-utilization.jsonl` 和 `steady-utilization-summary.json`。
四路并发显存接近上限且收益很小，采用三路。

## 启动与继续

```bash
PYTORCH_ALLOC_CONF=backend:cudaMallocAsync \
.cache/cacherepair-qwen-fast-env/bin/python -u scripts/RunCacheRepairTrainingFast.py \
  --job outputs/cacherepair-qwen38/full-training-repaired-v5 \
  --model /data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B \
  --dataset outputs/cacherepair-qwen38/full-training-v1/data \
  --config methods/CacheRepair/qwen38/configs/Qwen3.8-27B-Full.json \
  --pipeline-depth 3
```

意外中断后使用相同命令恢复。已建立训练目录时，监督器转发 `--resume auto`，不会重复导入统计。
手动停止后可加 `--physical-gpus 0,1,2,3` 或 `--physical-gpus 0,1`，恢复为四卡或两卡。
代码、数据 manifest 和配置被固定；变更数学计算不属于普通 resume。
分配器环境变量随启动进程继承，恢复时也应保留上述 `PYTORCH_ALLOC_CONF` 设置。

v5 已包含原子 RMS 提交、orphan checkpoint 恢复、epoch 完成标记、共享官方 model.py 哈希保护，
以及合并梯度检查标量读取。主命令行会在发布资源就绪文件前建立输出父目录。
源码切换需要明确审计证据；迁移只更新身份元数据，序列化后逐项验证模型、optimizer、RNG 和
游标保持一致。禁止为绕过不兼容而手工关闭 identity 检查。
