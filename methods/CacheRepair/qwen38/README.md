# Qwen3.8-27B CacheRepair-M 模型本体

本文说明模型、缓存坐标边界和 checkpoint 接口。后续增加的数据准备、去重和官方训练
循环适配见 [TRAINING.md](TRAINING.md)。在线 GDN TRR + learned KV 组合见
[HybridCacheRepair](../../HybridCacheRepair/README.md)。
已完成的工程烟测见 [STAGE_A_REPORT.md](STAGE_A_REPORT.md)。
256/32 数据、1 epoch 的训练及固定集验证结果见 [STAGE_B_REPORT.md](STAGE_B_REPORT.md)。
原有 Qwen2.5 `CacheRepairMethod` 继续使用原来的实现与权重。
冻结目标的 vLLM KV 生成、CUDA IPC、多卡预取与性能短测见 [VLLM_TRAINING.md](VLLM_TRAINING.md)。

## 来源与容量选择

网络直接复用仓库中未修改的官方 `StaleEncoder`、`RepairBlock` 和其中的 SDPA
attention，来源为 [CacheRepair-code](https://github.com/genglinWang/CacheRepair-code)，
vendored commit 为 `4eb96cd92df4be4f679ad8ac42a51442249683c2`。
此次核对的官方 `model.py`、`cache.py`、训练说明及 Qwen2.5-3B/14B-M 配置与 vendored
文件逐字节相同。许可证与归属见 `../backend/LICENSE`、`../backend/NOTICE`。

本版本采用任务说明中的宽度 640、6 blocks、10 heads、segment width 16、MLP ratio
3.0 和内部 RoPE theta 10000，与官方
[Qwen2.5-14B-M 配置](https://github.com/genglinWang/CacheRepair-code/blob/main/configs/Qwen2.5-14B-Instruct-M.json)
完全一致。Qwen2.5-3B-M / Llama-8B-M 则使用宽度 512、8 heads；M 的具体宽度随
目标不同。这里根据 Qwen3.8 的实际几何改变 encoder 的 segment 数和 residual head
输出维数。
网络结构保持官方设计，没有增加独立 K/V 网络、额外 encoder MLP 或新的条件分支。

## 本地目标核对

检查的 checkpoint：

```text
/data1/lyh/.cache/modelscope/hub/models/Qwen/Qwen3.8-27B
```

它的原生 architecture 是 `Qwen3_5ForConditionalGeneration`，文本主干为
`model.language_model`，model type 为 `qwen3_5_text`。
本地配置与 [Qwen 官方配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
的相关几何一致；实现从传入目标模型的配置和实际模块推导尺寸，不依赖这张表：

| 项目 | 本地值 |
| --- | --- |
| 文本层数 / hidden size | 64 / 5120 |
| 全注意力层（从 0 开始） | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 |
| Attention Q heads / KV heads / head dim | 24 / 4 / 256 |
| Target rotary dim / theta | 64 / 10,000,000 |
| Target RoPE | default, partial 0.25, interleaved MRoPE [11, 11, 10] |
| Target attention scaling | 1.0 |
| 文本词表 / embedding dim | 248320 / 5120 |
| 配置最大位置数 | 262144 |

还核对了本地 safetensors 的 embedding `[248320,5120]`、K/V projection
`[1024,5120]` 和全注意力权重的层索引。checkpoint 另有 MTP 权重，但它不属于文本
主干的 64 个 decoder layers，不计入 repairer。

`TargetGeometry.from_target` 检查配置、模块类型、K/V projection 大小、embedding
归属以及原生 rotary 的频率与维数。`pack_target_kv` 再检查运行时 hybrid cache
的全层映射和实际 K/V 形状，拒绝把 recurrent state 当成 K/V。

## 模型与张量约定

```text
canonical KV [B,T,16,4,2,256]
  ├─ 每个 segment 独立 256→16 → concat 2048 → Linear 640 → LayerNorm
  └─ 共享 frozen target embedding 5120 → Linear 640
        concat 1280 → Linear 640
        6 × (零初始化 stale reinjection + 官方 AdaLN RepairBlock)
        LayerNorm → 零初始化 Linear 640→32768
        normalized residual [B,T,16,4,2,256]
```

每个 block 为 10-head attention，head dim 64；MLP 为 `640→1920→640`，
激活 SiLU，AdaLN 的 shift/scale/gates 从零开始。Repairer 内部 RoPE theta 为
10,000，与目标模型的 RoPE 独立。

模型参数量为 **54,017,664**，FP32 参数约 **206.06 MiB**，两个 FP32 RMS buffer
合计 256 KiB，不含目标模型及共享 embedding。Canonical FP32 KV 每 token
128 KiB；4096 tokens 的单份 KV 约 512 MiB。

唯一 segment 顺序为 `attention layer → KV head → K/V → dimension`，K/V 轴
0 为 K、1 为 V。输入 encoder 和输出 head 共用此顺序。

`forward(stale_kv, token_ids, chunk_ids)` 输出归一化 residual：

```python
stale_norm = stale_kv.float() / sigma_stale.float()
delta_norm = repairer(stale_kv, token_ids, chunk_ids)
repaired = stale_kv.float() + delta_norm.float() * sigma_delta.float()
```

`sigma_stale`、`sigma_delta` 均为 `[L_attn,H_kv,2,D]` 的 buffer，初值为 1。
`set_normalization` 用于安装后续外部估计的逐坐标 RMS，检查形状、有限性和正值。
统计量由训练入口估计并保存；新建模型尚未安装训练数据统计量，初始模型没有已训练的修复能力。

Mask 精确遵循 `query_chunk >= key_chunk`，同 chunk 双向、前 chunk 可见、后 chunk
不可见。`chunk_ids=-1` 表示 padding；padding query 只看自己，padding KV 在投影前
清零，padding 的 residual 输出为零。`token_ids` 和 `chunk_ids` 的形状为 `[B,T]`。
Repairer 内部位置为当前完整拼接输入的 `arange(T)`；不要把它分割成独立窗口调用。

## 原生缓存与 RoPE 边界

`pack_target_kv` 提取的是目标 K-norm、RoPE 之后的 K，以及原生 V。
`canonicalize_target_kv`、`materialize_target_kv` 调用目标原生 rotary 模块生成
FP32 cos/sin，支持文本 `[B,T]` 和原生三轴 `[3,B,T]` position IDs。
目标 text model 的四轴 position IDs 中第一轴供 causal mask 使用；传给这些
RoPE 函数时应取后三轴，与原生 rotary 调用保持一致。

只有前 64 个 K 维度旋转，其余 K 坐标与 V 保持原值（canonical 转 FP32，
materialization 最后统一转 cache dtype）。逆变换包含 cos²+sin² 的除法，
正确处理 attention scaling。BF16 原生 cache 已经存在舍入误差，canonical
变换不会恢复缓存前被舍弃的精度。

`reposition_target_kv` 用 local/global 原生 cos/sin 计算 delta rotation，仅移动
K，并先舍入回原 cache dtype；之后再按 global positions canonicalize，保留官方
stale construction 的舍入阶段。`supervision.py` 在其上执行独立 chunk prefill 和
真实 joint mixed-model forward。只支持此目标实际使用的 default RoPE；动态频率变体明确拒绝。

`replace_target_kv` 只在**已有完整 hybrid cache** 中替换同形状的全注意力 K/V，
保持 K/V tensor 对象、cache 长度信息和所有 GDN states。仅有 repaired attention
KV 不能组成完整可生成的 hybrid cache；后续在线接入必须提供既有 GDN state
路径。不能套用旧 Qwen2.5 的 `to_hf_cache` 将 16 个 attention layers 当作全部层。

## Python API

以下 `target` 是由支持 Qwen3.5 的 Transformers 加载的实际目标模型；`native_past`
是它已有的完整 hybrid cache，`positions` 是生成该 cache 时使用的实际位置。

```python
from methods.CacheRepair.qwen38 import (
    QwenCacheRepair, pack_target_kv, canonicalize_target_kv,
    materialize_target_kv, replace_target_kv, text_backbone,
    save_repairer, load_repairer,
)

model_id = "Qwen/Qwen3.8-27B"
repairer = QwenCacheRepair.from_target(target, target_model_id=model_id)
repairer.to(device=target.get_input_embeddings().weight.device)  # repairer 默认 FP32
rotary = text_backbone(target).rotary_emb
native = pack_target_kv(native_past, repairer.target_config)
canonical = canonicalize_target_kv(native, positions, rotary)
fixed = repairer.repair(canonical, token_ids, chunk_ids)
materialized = materialize_target_kv(fixed, positions, rotary, dtype=native.dtype)
replace_target_kv(native_past, materialized, repairer.target_config)

save_repairer("outputs/qwen38-repairer", repairer)
loaded = load_repairer("outputs/qwen38-repairer", target, target_model_id=model_id)
```

Factory 和 loader 将目标冻结并切为 eval。共享 embedding 不是 repairer 注册的
子模块，`repairer.to()`、`parameters()`、`state_dict()` 不会移动、转换或重复
保存它；目标 embedding 和 repairer 输入需在同一 device。直接使用构造函数时，
调用方负责冻结 embedding 之外的目标模型。本轮 API 面向单 device repairer。

Checkpoint 使用独立格式 `kvbench-qwen-cacherepair-v1`，记录目标 ID、文本配置
指纹、全注意力层映射、KV 几何、原生 RoPE 参数/频率指纹、架构配置、坐标与
segment 顺序，并保存两个 RMS buffers。加载严格检查这些元数据和 state dict，
不兼容的 checkpoint 会被拒绝。使用显式规范 ID 可让相同 checkpoint 的不同本地
路径共用 repairer；配置指纹不是目标权重内容的哈希。

## 验证与环境

项目原环境为 Transformers 4.57.6，不能实例化此 Qwen3.5 主干。
本次在 `.cache/cacherepair-qwen-env` 的隔离环境中安装了 Transformers 5.8.0
做原生验证，没有修改共享项目依赖。

```bash
# 模型本体与现有缓存/提示词/Transformers 方法的相关回归
python -m pytest -q tests/methods/test_cacherepair_qwen38.py \
  tests/backends/test_cache_backend.py tests/backends/test_prompt.py \
  tests/backends/test_transformers.py tests/backends/test_model_adapter.py \
  tests/methods/test_naive.py tests/methods/test_full_prefill.py

# 原生 Qwen3.5 hybrid cache / partial MRoPE 验证
.cache/cacherepair-qwen-env/bin/python -m pytest -q \
  tests/backends/test_cacherepair_qwen38_native.py
```

模型测试包含：完整尺寸 M 前向与零 residual 恒等、非零权重下与官方网络对齐、
真实 chunk 因果依赖、padding 中 NaN/非法 token 的隔离、逐坐标缩放、segment
顺序、embedding 所有权、checkpoint round trip 和不兼容拒绝。
原生测试使用随机初始化的小型 Qwen3.5，正常执行 GDN + attention，验证 MRoPE
三轴、partial RoPE、FP32/BF16 round trip、非单位 scaling、chunk reposition 和
替换 K/V 后继续 decode。

验证结果：项目环境中模型与相关回归 **85 passed**；隔离环境中模型与原生检查
**21 passed**（其中 13 项模型测试与前者重复，8 项为原生检查）。

本地 27B checkpoint 的额外核对：用其真实 config 在 meta device 构造完整原生
architecture，从实际 safetensors 共享 BF16 embedding，以 synthetic canonical KV
执行完整尺寸 M 前向。输出 `[1,6,16,4,2,256]`，零 residual 恒等通过。
原始检查结果见工作区 `.cache/cacherepair-qwen38-local-inspection.json`。
没有执行完整 27B 权重的目标前向；真实 hybrid cache 测试来自上述小模型。

上面的验证记录对应最初的模型本体实现；训练准备和 Stage A 记录见 [TRAINING.md](TRAINING.md)。
尚无在线质量/TTFT 结果。
SDPA 精确 mask 是正确性基线，mask 存储随 T² 增长，尚未针对长上下文优化。
训练入口已提供 RMS 估计、stale/joint pair 和 checkpoint 验证。
`HybridCacheRepair` 在独立的 serving adapter 中完成 Hypic GDN state 与修复 KV 的
组合、TP 与长前缀分块推理，并记录完整在线 TTFT；具体效果以对照评测为准。

## 文件

| 文件 | 职责 |
| --- | --- |
| `model.py` | M 配置、官方组件组装、padding、normalized residual、checkpoint |
| `target.py` | 实际文本主干识别、层映射与兼容性元数据 |
| `cache.py` | 全注意力 KV 提取/替换、原生部分 MRoPE 坐标转换 |
| `configs/Qwen3.8-27B-M.json` | 新目标容量配置，不预填目标几何 |
| `__init__.py` | 独立 Python API |
| `tests/methods/test_cacherepair_qwen38.py` | 模型、mask、embedding 与存储测试 |
| `tests/backends/test_cacherepair_qwen38_native.py` | 原生 Qwen 缓存与 RoPE 测试 |
