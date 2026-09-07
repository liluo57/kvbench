# 技术报告二：AWS Hybrid Model Factory 的 Hybrid State Composition

## 0. 报告目的

本文描述 AWS `awslabs/hybrid-model-factory` 当前公开代码中的 State Composition 实现。

重点回答：

```text
Hybrid model 中：
Full Attention cache 怎么处理？
Gated DeltaNet state 怎么处理？
Prepare/Compose/Query 整个 pipeline 怎么走？
```

以及公开实现中存在的一个重要缺口：

```text
GDN PICASO fuse 的 projection_matrix 消费端已经实现，
但当前公开代码中没有找到对应的生产端。
```

---

# 1. AWS 要解决的问题

AWS 的目标是把一个长 context 切成多个 chunk：

```text
C1
C2
C3
...
```

每个 chunk 独立进行 prefill，然后把 hybrid model 中两类 memory：

```text
Full Attention -> KV cache
SSM/GDN       -> recurrent state
```

重新组合成一个 HybridCache。

最后再从这个 composed cache 开始 forward query。

StateComposition 文档将方法描述成：

```text
context
   ↓ split
chunk 1 ── forward ── cache 1
chunk 2 ── forward ── cache 2
chunk 3 ── forward ── cache 3
   ↓
layer-wise compose
   ↓
HybridCache
   ↓
prefill query
   ↓
generation
```

文档推荐的默认配置是：

```text
concat_kv_soup_ssm
sequential_positions=False
shared prefix
query prefilled after composition
```

---

# 2. AWS 实现了九种 compose_type

方法被拆成两个独立维度。

## KV strategy

```text
full
concat
sw
```

## SSM strategy

```text
kv_only
soup
fuse
```

组合成：

| compose_type         | Full Attention        | SSM/GDN           |
| -------------------- | --------------------- | ----------------- |
| `full_kv_only`       | full KV               | zero state        |
| `full_kv_soup_ssm`   | full KV               | mean              |
| `full_kv_fuse_ssm`   | full KV               | fuse              |
| `concat_kv_only`     | independent KV concat | zero state        |
| `concat_kv_soup_ssm` | independent KV concat | mean              |
| `concat_kv_fuse_ssm` | independent KV concat | PICASO-style fuse |
| `sw_kv_only`         | sliding-window KV     | zero              |
| `sw_soup_ssm`        | sliding-window KV     | mean              |
| `sw_fuse_ssm`        | sliding-window KV     | fuse              |

代码中的 `compose_type_map` 就是这样映射。

---

# 3. 对真正 reuse 最相关的是 concat_kv 系列

AWS 的 `full_kv_*` 并不是严格意义上的 offline context reuse。

它会把：

```text
C1 || C2 || C3
```

重新做一次完整 forward，从而得到真实完整 KV cache；

然后只把 SSM state 换成 composed state。

所以：

```text
full_kv_*
```

更适合作为 quality/reference experiment。

真正接近“chunk 已经 Prepare，以后不再重新 forward”的是：

```text
concat_kv_*
```

其核心过程是：

```text
C1 -> independent forward
C2 -> independent forward
C3 -> independent forward

FullAttn:
    KV1 || KV2 || KV3

GDN:
    compose(State1, State2, State3)
```

AWS `_compose_batch_hybrid()` 正是这样做的：把 chunks 独立作为 batch 进行 model forward，再逐层建立一个新的 HybridCache。

---

# 4. Chunk 是怎样独立 forward 的

AWS 会：

```text
1. split context
2. 给每个 chunk prepend 一个共享 prefix
3. pad 到相同长度
4. 把多个 chunk 放在 batch dimension
5. 一次 model forward
```

如果 batch prefill OOM，则 fallback：

```text
for chunk:
    independently forward chunk
    merge cache along batch dimension
```

因此算法语义上始终是：

```text
每个 chunk 独立 prefill
```

并不是：

```text
C1 forward 后把 state 传给 C2
```

---

# 5. Position IDs

AWS 支持：

```text
sequential_positions=True
```

或者：

```text
sequential_positions=False
```

推荐的是：

```text
False
```

这意味着所有 independent chunks 都使用自己的 local position：

```text
chunk1: 0 1 2 3 ...
chunk2: 0 1 2 3 ...
chunk3: 0 1 2 3 ...
```

而不是：

```text
chunk1: 0 ...
chunk2: L ...
chunk3: 2L ...
```

这主要影响 Full Attention 层中的 RoPE cache。

GDN 本身没有 RoPE position dependency，但它收到的 hidden states 会受到前面 hybrid layers 影响。

---

# 6. Full Attention：concat KV

对于 attention layer，每个 independent chunk 得到：

```text
K_i
V_i
```

AWS 将这些 cache 在 sequence dimension 上拼接。

由于每个 chunk 前面可能重复包含 shared prefix/padding，因此它有 `prefix_masks`，真正 concat 前会删除重复 prefix/padding cache。

逻辑等价于：

```text
K = cat(
    valid(K_chunk1),
    valid(K_chunk2),
    ...
)

V = cat(
    valid(V_chunk1),
    valid(V_chunk2),
    ...
)
```

代码函数为：

```text
concat_kv_with_mask(...)
```

### 重要性质

这些 K/V 是：

```text
chunk independently forward
```

产生的。

所以它和：

```text
forward(C1 || C2 || C3)
```

得到的真实 K/V 并不相同。

AWS 没有做 CacheBlend 那种 KV repair。

因此 `concat_kv` 本质是 **naive independently-prefilled KV concatenation**。

---

# 7. GDN cache：AWS 假定每个 chunk 有什么

AWS `_merge_chunked_cache()` 期待每个 GDN layer 的 cache entry 至少可以包含：

```text
recurrent_state
conv_state
projection_matrix    # fuse 时需要
```

多个 chunk 独立 forward 后，这些量沿 batch dimension 堆起来：

```text
recurrent_states:
    [chunk1,
     chunk2,
     chunk3,
     ...]

projection_matrices:
    [chunk1,
     chunk2,
     chunk3,
     ...]

conv_states:
    [chunk1,
     chunk2,
     chunk3,
     ...]
```

---

# 8. GDN 方法一：kv_only

最简单的 ablation：

```text
Full Attention KV 保留
GDN recurrent state = 0
GDN conv state      = 0
```

其目的就是测试：

> Hybrid model 的信息是否可以主要靠 Full Attention KV retrieval，而完全忽略 GDN memory。

代码中 `zero_gated_deltanet_states()` 会同时 zero recurrent state 和 convolution state。

---

# 9. GDN 方法二：Soup

这是 AWS 当前最简单、实现最完整、文档也明确推荐的路线。

设每个 independently processed chunk 得到：

$$
H_i.
$$

直接：

$$
H_{compose}
=
\frac1N\sum_iH_i.
$$

代码：

```text
recurrent_states
    -> mean(dim=chunk/batch)
```

对于 GDN 的 convolution history：

$$
C_{compose}
=
\frac1N\sum_iC_i.
$$

如果 conv state 有多个 component，则每个 component 分别 mean。

换言之：

```text
recurrent state -> average
conv state      -> average
```

完全没有 transition-aware correction。

这就是：

```text
concat_kv_soup_ssm
```

里的 GDN 部分。

---

# 10. AWS 对 Soup 的公开结果

AWS 文档明确把：

```text
concat_kv_soup_ssm
```

作为推荐设置：

```text
independent chunk KV concat
+
SSM/GDN state average
```

其 NIAH example 声称，在发布的 8B instruction-tuned hybrid models 上：

```text
Mamba2
Gated DeltaNet
Gated Kalman Net
```

使用该方案最多到 8 chunks 都取得了 100% retrieval accuracy；GKA 在 16 chunks 时约 80%。

这只是其 NIAH example 结果，不应该理解成“九种方法全部经过完整 benchmark 后 Soup 全面胜出”。

---

# 11. GDN 方法三：AWS 的 PICASO-style Fuse

AWS 为 GDN 写了：

```text
picaso_combine_gated_deltanet()
```

它要求每个 independently processed chunk 有：

```text
H_i = recurrent_state_i
P_i = projection_matrix_i
```

然后：

```text
coeffs = get_matrix_coef_for_picaso(P_1, ..., P_N)

H_comp =
    Σ coeff_i @ H_i
```

与此同时：

```text
conv_state
```

依然只是简单平均。

即：

```text
recurrent_state -> PICASO matrix composition
conv_state      -> mean
```

---

# 12. AWS 的 get_matrix_coef_for_picaso 做了什么

其实现明显来自原 PICASO-S 的 elementary-symmetric-polynomial DP。

对第 idx 个 chunk：

```text
1. 删除 P_idx
2. 对其余 matrices 做 DP
3. 构造不同阶数的 matrix products
4. 使用 1 / C(N-1, m) weighting
5. 再除以 N
6. 得到 coefficient W_idx
```

DP recurrence 在 scalar/diagonal 情况类似：

$$
e_m^{(j)}
=
e_m^{(j-1)}
+
A_j e_{m-1}^{(j-1)}.
$$

AWS 的 matrix 版本直接改成：

$$
E_{m,j}
=
E_{m,j-1}
+
A_j @ E_{m-1,j-1}.
$$

最后：

$$
H_{compose}
=
\sum_iW_iH_i.
$$

---

# 13. 这里和原始 PICASO 有一个重要理论区别

原 PICASO-S 的 Proposition 2 在推导该 symmetric-polynomial closed form 时明确假设：

$$
A_iA_j=A_jA_i.
$$

Mamba 中 A 为 diagonal，因此成立。

GDN 的：

```text
P_i
```

则是 dense matrix，一般：

$$
P_iP_j\neq P_jP_i.
$$

AWS 的 `get_matrix_coef_for_picaso()` 直接把 scalar multiplication 换成：

```text
@
```

matrix multiplication。

问题是它的 DP 对一个 matrix subset 只产生一个固定 order 的 product，而真正的 full-permutation average 在非交换矩阵情况下还涉及同一 subset 的不同 matrix ordering。

因此：

> AWS 的 GDN matrix `fuse` 应理解为对 PICASO-S 的工程 extension，而不是原论文 Proposition 2 对 GDN 的直接严格应用。

这一点在复现时不要误认为是理论等价。

顺带值得注意：原始 PICASO-R 的推导只要求 transition matrix 可逆，而不要求 commute。因此如果未来专门研究 GDN transition composition，matrix PICASO-R 是另一条很自然的路线；但 AWS 当前这里实现的是 matrix PICASO-S-style DP，而不是 matrix PICASO-R。

---

# 14. 更重要的问题：AWS 公开代码的 GDN Fuse 路径似乎没有接完整

`picaso_combine_gated_deltanet()` 明确要求：

```text
cache[layer]["projection_matrix"]
```

如果没有，会直接：

```text
raise ValueError
```

而 `_compose_batch_hybrid()` 在 `fuse` 模式下也 assert：

```text
GDN cache 中必须出现 projection_matrix
```

但是检查当前公开仓库发现：

```text
projection_matrix
```

在整个 repository code search 中只出现在：

```text
cache_compose.py
```

也就是消费端。没有找到 GDN forward 中生产它的实现。

当前 HMF GDN forward 正常写入 cache 的是：

```text
recurrent_state
conv_state
```

而 `GatedDeltaNetCache.update()` 本身也只有这两个 state 字段，没有 `projection_matrix` 参数。

因此当前 public main 呈现的是：

```text
GDN forward
    ↓
recurrent_state
conv_state

           projection_matrix ???
                    ↓
picaso_combine_gated_deltanet()
```

也就是说：

> **GDN soup 路径是完整闭环的；GDN fuse 路径的 composition code 存在，但当前公开代码缺少 projection_matrix 的生产部分。**

复现时 Codex 不应该花时间寻找一个实际不存在的 `projection_matrix` cache hook；需要自行补上这个计算。

---

# 15. Qwen3.8 GDN 中 projection_matrix 应代表什么

这一节是为了说明 AWS 消费端期待的数据语义，不是声称 AWS 已经实现。

Qwen3.8 使用的 GDN recurrent update 可以写成：

首先 decay：

$$
\tilde S_t=e^{g_t}S_{t-1}.
$$

然后 delta rule：

$$
m_t=k_t^\top\tilde S_t
$$

$$
\delta_t=\beta_t(v_t-m_t)
$$

$$
S_t=\tilde S_t+k_t\delta_t^\top.
$$

HuggingFace 当前 Qwen GDN recurrent implementation 就是按这种形式更新 state。

整理：

$$
S_t
=
e^{g_t}
(I-\beta_tk_tk_t^\top)
S_{t-1}
+
\beta_tk_tv_t^\top.
$$

定义：

$$
P_t=
e^{g_t}
(I-\beta_tk_tk_t^\top).
$$

则一个 token 是 affine transform：

$$
S_t=P_tS_{t-1}+B_t.
$$

整个 chunk：

$$
S_{out}
=
P_{chunk}S_{in}+H_{chunk}
$$

其中：

$$
P_{chunk}
=
P_TP_{T-1}\cdots P_1.
$$

而：

$$
H_{chunk}
=
F_{chunk}(0)
$$

就是普通 independent prefill 已经返回的 final recurrent state。

因此 AWS 所谓：

```text
projection_matrix
```

最自然的语义就是这个：

$$
\boxed{P_{chunk}}
$$

即该 chunk 对任意 incoming recurrent state 的累计左侧 transition。

---

# 16. Qwen3.8 上计算 P_chunk 时的关键点

若 Codex 实现 AWS GDN fuse，必须从 Qwen GDN 实际 kernel 使用的量计算，而不能随意从 raw hidden states 近似。

每个 token 应得到：

```text
k_t
beta_t
g_t
```

然后：

$$
P_t=e^{g_t}(I-\beta_tk_tk_t^\top).
$$

之后按 token 顺序：

```text
P_chunk = P_t @ P_chunk
```

初始：

```text
P_chunk = I
```

最终和：

```text
H_chunk = final recurrent state
```

一起缓存。

尤其需要使用 **GDN recurrence 真正使用的 normalized/expanded key**。

Qwen 的 delta-rule kernel 对 Q/K 有自己的 normalization 逻辑，因此不能直接拿 projection layer 尚未 normalization 的 raw K 来构造 transition。

---

# 17. Conv state：AWS 没有精确组合，只是 mean

Qwen GDN 在 recurrent matrix 之前还有 causal convolution。

因此 continuation query 不仅依赖：

```text
recurrent_state
```

也依赖：

```text
conv_state
```

Qwen3.5/Qwen3.8 的 GDN implementation 确实在 cached decode 中读取之前的 convolution state。

AWS 无论：

```text
soup
```

还是：

```text
PICASO fuse
```

对 convolution state 都只做：

$$
C=\operatorname{mean}(C_i).
$$

所以 AWS 并没有给 convolution boundary composition 一个严格推导。

这是明确的 heuristic。

复现 AWS 时就应该照着：

```text
recurrent_state:
    soup -> mean
    fuse -> PICASO matrix combination

conv_state:
    mean
```

不要额外假定存在更复杂的 conv repair。

---

# 18. Compose 后怎么处理 query

AWS 不是 composition 完成后直接 generate 第一 token。

它先：

```text
context chunks
     ↓
compose HybridCache
     ↓
query tokens
```

将 suffix/query token 正常 forward 到 composed cache 中。

其 `prefill_query()` 会逐 token 使用：

```text
past_key_values = composed_cache
```

继续更新：

```text
Full Attention KV
GDN recurrent state
GDN conv state
```

完成 query prefill 后才进入 generation。

因此对应我们的 reuse 系统应该明确区分：

```text
reused document tokens:
    0 model forward at runtime

new query tokens:
    normal forward
```

---

# 19. AWS 整个 concat_kv_soup_ssm pipeline 可浓缩为

```text
PREPARE / independent prefill
=============================

for chunk Ci:

    Forward(Ci, zero/empty cache)

    for FullAttn layer:
        save Ki, Vi

    for GDN layer:
        save recurrent_state Hi
        save conv_state Ci


COMPOSE
=======

for FullAttn layer:
    K = concat(K1, K2, ...)
    V = concat(V1, V2, ...)

for GDN layer:
    H = mean(H1, H2, ...)
    C = mean(C1, C2, ...)


QUERY
=====

install composed HybridCache

Forward(query_tokens, composed_cache)

Generate(...)
```

这是目前把 AWS 方法应用到 Qwen3.8 最容易完整复现的一条路线。

---

# 20. AWS concat_kv_fuse_ssm pipeline

理论上的 intended pipeline 是：

```text
PREPARE
=======

for each chunk:

    FullAttn:
        K_i
        V_i

    GDN:
        H_i = recurrent final state
        C_i = convolution final state
        P_i = projection / cumulative transition


COMPOSE
=======

FullAttn:
    concat independent K/V

GDN:
    W_i = get_matrix_coef_for_picaso(P_1 ... P_n)

    H = Σ W_i @ H_i

    C = mean(C_i)


QUERY
=====

load composed hybrid cache

forward query only
```

其中当前 AWS public code 已经有：

```text
get_matrix_coef_for_picaso
picaso_combine_gated_deltanet
```

但缺：

```text
在 GDN prefill 中得到 P_i 的 producer
```

所以复现到 Qwen3.8 上必须补这个环节。

---

# 21. 给 Qwen3.8 Method 实现的最小数据契约

如果目标是让后续可以同时实现 Soup 和 Fuse，Prepared chunk 最好至少抽象为：

```text
PreparedChunk

for each Full Attention layer:
    key_cache
    value_cache

for each GDN layer:
    recurrent_state
    conv_state

    optional:
        projection_matrix
```

其中：

```text
Soup:
    不需要 projection_matrix

Fuse:
    必须需要 projection_matrix
```

也就是说 Prepare 第一版完全可以先完成：

```text
KV
recurrent_state
conv_state
```

并验证：

```text
concat_kv_soup_ssm
```

然后第二阶段再增加：

```text
projection_matrix
```

而不需要推翻 PreparedChunk 格式。

---

# 22. 推荐的复现顺序——这是对 AWS 方法的工程拆解，不是新算法

### Phase A：AWS Soup

实现：

```text
FullAttn:
    independent KV concat

GDN:
    recurrent state mean
    conv state mean
```

对应：

```text
concat_kv_soup_ssm
```

目的：

```text
确认 Qwen3.8 的 cache extraction
确认 cache injection
确认 GDN state continuation
确认 hybrid composed cache 能正常 forward query
```

### Phase B：AWS GDN Fuse

在 Prepare 时额外记录：

```text
P_chunk
```

然后原样移植：

```text
get_matrix_coef_for_picaso
```

以及：

```text
Σ W_i @ H_i
```

conv state 仍然 mean。

### Phase C：再评估理论/算法问题

只有在 Phase B 可跑后，再讨论：

```text
AWS matrix PICASO-S 是否合理
PICASO-R 是否更适合 non-commuting GDN transitions
Full Attention KV 是否需要 CacheBlend repair
conv state 是否需要 repair
```

这些不属于 AWS 当前方法本身，不应该混进第一轮“复现 AWS”。
