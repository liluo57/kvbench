# Qwen3.8 冻结目标的 vLLM 监督后端

`train.py --supervision-backend vllm` 使用 `VllmPairPool` 生成 stale/joint KV。
修复器结构、FP32 参数、原来的归一化残差 MSE、每 4 个样本梯度累积一次、
AdamW、样本顺序和检查点协议保持原设计。此次混合数据微调继续继承 epoch 3 的 RMS。

## 边界与数据路径

- 每个冻结目标是一个独立的 vLLM TP=2 实例；全部副本都在独立进程中运行。
- 权重、attention KV 为 BF16，GDN recurrent state 为 FP32；不量化。
- eager、关闭 prefix caching、关闭 chunked prefill、关闭 async scheduling。
  每个独立文档 chunk 从空 attention cache 和空 GDN state 开始，joint 请求使用完整拼接输入。
- 只从 16 个 full-attention 层的 cache writer 输入捕获 K/V。逐请求核对输入 token IDs、
  完整 fresh prefill、层集合、TP rank 和 KV head 次序；不读取分页缓存的 padding。
- 原生 BF16 K/V 经 CUDA IPC/P2P 直接传到 repairer GPU。CPU 只传元数据，KV 不落盘。
  消费进程先完成复制，再显式通知 producer 释放引用；pair-local CUDA ordinal 映射到
  消费进程看到的 GPU 编号，支持 `CUDA_VISIBLE_DEVICES` 重映射。
- 在 GPU0 按原有协议进行 chunk rerotation、BF16 rounding、FP32 inverse RoPE。
  预分配输出并分块计算，保留 `[1,T,16,4,2,256]` FP32 stale/joint 接口。
- 修复器读取同一 checkpoint 的冻结 BF16 embedding。只加载这张 embedding 表到训练进程，
  不再在训练进程额外加载整套 27B。

这次后端切换不声称与 native 逐位相同。此前 token、算子与误差调查见
`outputs/cacherepair-qwen38/vllm-difference-investigation-20261028/REPORT.md`。
此次 P2P 数据完整性以 producer/consumer 原始 BF16 字节 SHA256 相等验证；它证明传输
没有改变捕获值，不证明 vLLM 与 native 数值相同。

## 显存、队列与停止恢复

必须使用 `PYTORCH_ALLOC_CONF=backend:native`，CUDA IPC 不使用 cudaMallocAsync allocator。
每个 TP worker 固定保留 1 GiB 的分页缓存预算，避免启动内存 profiling 的波动使 16K
可用长度不足，或者把共享 GPU0 留给修复器的空间吃完。

`--vllm-prefetch-per-worker` 控制远端副本的有界预取；与训练共享 GPU 的副本最多保留
一条 pending pair。`--vllm-scheduler least_work` 根据尚未完成的输入 token 工作量分配副本，
消费和 optimizer accumulation 仍严格按原始样本顺序。已完成的 KV 不计作待计算工作。
`--vllm-prefetch-limit` 单独限制全局pending数量；最终采用远端上限8、共享组1、全局9，
避免每组相同的槽数限制把长样本堆到一组。复制完成后异步排队释放，在阶段切换/关闭时
核对所有ack，避免释放RPC排在多个prefill后时同步阻塞训练。

GPU0 与目标共享时，在申请新的目标 forward 前释放训练进程的空闲 allocator blocks，
在释放 producer KV 后释放共享 rank 的空闲 blocks。对于至少 8192-token 的样本，
在 KV 转换前、长序列反向前等待同卡目标 prefill 完成并释放其空闲 blocks，避免两个
独立进程的显存峰值叠加。其他 GPU 副本继续执行。

`RunCacheRepairTrainingFast.py` 转发后端、预取、调度和显存参数，继续使用原来的
每小时 500 例 MSE、每 epoch 固定 10-task×64 质量评测。Full/TRR 历史对照继续复用。
`--dedicated-repairer` 在 6/4 张卡时将逻辑 GPU0 留给训练，使用 2/3（以及 4/5）推理；
在两卡恢复时自动回到 0/1 共享。此模式会有一张未使用的逻辑 GPU1。
重新分配 GPU 不改变修复器/AdamW/数据游标的恢复格式。

## 不保存状态的性能短测

`scripts/ProfileCacheRepairTraining.py` 从指定 checkpoint 加载修复器与 RMS，在真实数据上
执行完整的 forward、backward、clip 和 AdamW update，写入逐样本/逐阶段耗时及 1 秒采样
的 GPU 指标。它不会调用 `save_repairer`、保存 optimizer/RNG/data cursor，或创建正式
训练作业。试验结束后更新直接丢弃；源 checkpoint SHA256 和 RMS buffers 必须保持不变。

`--sweep` 复用已加载的冻结目标比较共享/专用 GPU0、队列、传输方式和修复器 token tile。
每个配置从同一个 checkpoint 重置，首轮预热、后续轮计时；`--operator-profile` 额外对
一个 16K 样本采集冻结目标和修复器的算子耗时。CPU host transport 只用于诊断对照。

当日实测结果及失败试验的边界说明见
`outputs/cacherepair-qwen38/vllm-training-profile-20261028/REPORT.md`。

## 跨训练样本 batch（2026-10-29 可选实验）

单个样本的 chunks/joint 原本已经共同提交，但多个训练样本需要额外合批。
`--vllm-batch-examples` 与 `--vllm-batch-input-tokens` 设置冻结目标的合批上限。
这里 input tokens 是训练输入T，目标实际forward还包括独立chunks与joint，约2T。
单条超过合批token预算的长样本会完整保留、单独处理。

`--vllm-remote-cache-gib`、`--vllm-remote-max-seqs` 与 `--vllm-max-batched-tokens`
只放大远端副本的资源：共享GPU0的副本仍为1 GiB、32 requests、最多32768 forward tokens、
每次1样本，保证其与repairer的显存共存。

`--vllm-dynamic-batching` 使用每个actor的调度线程，在上一个forward执行期间积累待处理
样本，actor空闲后将等待队列合批。不会先把每条小任务都提交到进程池后再尝试合并。
原始sample顺序、有效训练batch=4和梯度累积保持不变。

背压同时统计“正在生成”和“已生成但未消费”的输入token，只有GPU0消费后才退还额度。
`--vllm-worker-pending-tokens` 和 `--vllm-global-pending-tokens` 提供逐副本/全局限额，
另有 `--vllm-prefetch-limit` 的样本数限额。额度用尽后不再补交forward，生产不会无界增长。
大batch的各行通过同一BF16 allocation的不同CUDA IPC view传递，最后一行ack后才能释放
底层存储；阶段切换会排空生成与回收消息。

大batch会改变BF16数值结果，不能宣称与旧batch逐位相同。实验结果、首token和标签误差、
实际batch大小及待消费KV高水位见
`outputs/cacherepair-qwen38/vllm-batching-20261029/REPORT.md`。正式训练仍未启动。

## 单样本性能排查（2026-10-29）

默认继续每次一个训练样本。关闭动态合批时不创建 dispatcher 线程。
pair 日志分别记录训练进程清理 allocator 的 host 时间、长样本前后等待/trim RPC、
当前 future 等待、复制 event 的 host 等待；这些字段有嵌套关系，不能直接全部相加。
队列额度仍计训练输入 T，实际独立 chunks 与 joint prefill 合计约 2T；已完成但未消费的
pair 继续占额度，消费后的 producer IPC 引用在 release RPC 完成后才释放。

性能脚本的 `--single-sample-sweep` 比较原调度、长样本前延后共享组补队、共享组两槽三种
方案；`--cost-scheduler-sweep` 比较 token 工作量、attention 平方工作量及带副本速度权重
的诊断方案；`--finish-scheduler-sweep` 将新任务耗时也纳入完成时间估计。
这些方案在本次匹配对照中均没有稳定提速，只保留为可复现实验；生产默认仍为 token
工作量、共享组一槽，实验权重和完成时间估计关闭。
`--audit-kv-first-pass` 在每组预热轮核对所有 producer/consumer BF16 字节 SHA256，并将
各 rank 的摘要写入日志以便跨配置比较。稳态耗时排除预热轮；首 token 对比只覆盖各独立
prompt 的第一个 greedy token。

`scripts/ProfileCacheRepairCollectives.py` 使用两个新进程直接测量 vLLM PyNccl all-reduce，
在计时外检查常数求和及随机正负 BF16 的精确结果和字节摘要。通信协议需要在进程启动前
设置。NCCL 的 kernel entrypoint 名称不能单独证明实际执行协议。
本机 GPU5 当前链路为 Gen4 x8，其他五张为 x16；单独测得 4/5 的 all-reduce 带宽约为
0/1、2/3 的一半。排查和完整对照记录见
`outputs/cacherepair-qwen38/single-sample-optimization-20261029/REPORT.md`。
