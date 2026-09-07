# 技术报告一：PICASO —— Permutation-Invariant Context Composition with State Space Models

## 0. 报告目的

本文只描述原始 PICASO 论文的方法，不讨论 AWS Hybrid Model Factory 的后续实现。

论文：

**PICASO: Permutation-Invariant Context Composition with State Space Models**
Tian Yu Liu, Alessandro Achille, Matthew Trager, Aditya Golatkar, Luca Zancato, Stefano Soatto
ICLR 2025 / arXiv:2502.17605

PICASO 的目标是：

> 对若干已经离线处理过的 context，运行时不再把 context token 重新 forward 一遍，而是直接读取每个 context 对应的 SSM state，通过纯张量运算组成一个新的 state，然后从这个 state 开始 forward query。

论文称之为 **Database of States**。每个 context 不仅预计算一个 state，还需要额外保存描述该 context 状态转移的矩阵。这样在线组合多个 context 时不需要重新执行模型。

---

# 1. PICASO 的基本数学对象

考虑单层 input-dependent linear SSM：

$$
x_t=A(u_t)x_{t-1}+B(u_t)u_t.
$$

对于一个完整 chunk

$$
u=(u_1,\ldots,u_T),
$$

整个 chunk 可以看作从输入 state 到输出 state 的一个 affine transform：

$$
F_u(x)=A(u)x+x(u).
$$

这里：

$$
A(u)=A(u_T)A(u_{T-1})\cdots A(u_1)
$$

是整个 chunk 的累计 transition；

而

$$
x(u)=F_u(0)
$$

是该 chunk 从 zero initial state 开始 forward 后得到的 final state。

因此，**一个可复用 chunk 真正需要表达的是两个量**：

$$
\boxed{(x(u),A(u))}
$$

而不是只有 final state。

论文明确提出：离线数据库中预计算 context state，并额外保存对应的 weight/transition matrix。

可以把 Prepare 的抽象接口理解为：

```text
PreparedContext {
    state:      H = x(u)
    transition: A = A(u)
}
```

其中：

```text
H = chunk 从 zero state 独立 forward 得到的末态
A = chunk 对一个外部 initial state 的累计传播矩阵
```

---

# 2. CASO：先解决“怎么模拟 concatenation”

PICASO 首先定义了一个 order-dependent 方法：

**CASO — Compositional Aggregation of States as Observations**

假设有：

$$
u_1,u_2,\ldots,u_n.
$$

如果真的按这个顺序把 token 拼接起来：

$$
u=u_1u_2\cdots u_n,
$$

对于单层 SSM，最终 state 满足：

$$
x(u_1\cdots u_n)
=
x(u_n)
+
\sum_{i=1}^{n-1}
A(u_n)\cdots A(u_{i+1})x(u_i).
$$

这是论文 Proposition 1。对单层线性 input-dependent SSM，这是精确关系。

换一种对实现更直接的写法：

```text
S = 0

for context i in [1 ... n]:
    S = A_i @ S + H_i
```

其中：

```text
A_i = A(u_i)
H_i = x(u_i)
```

最后得到的 S，就是 CASO state。

因此，如果每个 chunk 在 Prepare 阶段已经缓存：

```text
(H_i, A_i)
```

在线运行时模拟：

```text
chunk1 -> chunk2 -> chunk3
```

根本不需要 forward chunk token，只需要：

```text
S = A1 @ 0 + H1
S = A2 @ S + H2
S = A3 @ S + H3
```

对于 Mamba，A 通常是 diagonal，所以这里甚至只是 element-wise multiplication，而不是 dense matmul。论文强调 CASO/PICASO 的在线 context composition 不需要模型 forward。

---

# 3. 为什么不能简单平均 state

假设两个 context A、B。

分别独立 Prepare：

$$
H_A=F_A(0)
$$

$$
H_B=F_B(0).
$$

真实顺序 A→B 的结果是：

$$
S_{AB}=A_BH_A+H_B.
$$

真实顺序 B→A 是：

$$
S_{BA}=A_AH_B+H_A.
$$

所以：

$$
H_A+H_B
$$

或者：

$$
(H_A+H_B)/2
$$

都没有模拟 SSM 的真实状态传播。

论文把直接平均 state 的方案作为 **Soup baseline**。

CASO 的改进就在于把每个较早 context 的 state 乘以后续 context 的 transition。

---

# 4. CASO 的问题：context 顺序

CASO 虽然模拟 concatenation，但明显 order-sensitive。

例如：

$$
S_{AB}=A_BH_A+H_B
$$

和：

$$
S_{BA}=A_AH_B+H_A
$$

一般不相等。

RAG 中检索出来的文档通常没有天然的时间顺序，因此 PICASO 的核心思路是：

> 不挑某一个 permutation，而是对不同排列得到的 CASO state 做平均。

定义：

$$
x^{PICASO}(u_1,\ldots,u_n)
=
\frac1{|G|}
\sum_{\pi\in G}
x^{CASO}(u_{\pi(1)},\ldots,u_{\pi(n)}).
$$

论文考虑两个 G：

1. 全排列群 \(S_n\)：**PICASO-S**
2. cyclic rotations \(C_n\)：**PICASO-R**

---

# 5. PICASO-S：平均所有 n! 种排列

PICASO-S 定义：

$$
G=S_n.
$$

也就是说理论目标是：

```text
for every permutation π:
    state_π = CASO(contexts ordered by π)

final_state = mean(state_π)
```

直接枚举显然是：

$$
O(n!)
$$

不可接受。

但 CASO 最终总可以整理成：

$$
S=\sum_i W_iH_i.
$$

因此问题变成：

> 不枚举所有 permutation，直接计算每个 chunk state \(H_i\) 应有的 coefficient \(W_i\)。

---

## 5.1 PICASO-S 的 coefficient

设除去第 k 个 chunk 后，其余 transition 为：

$$
A_{-k}=\{A_1,\ldots,A_{k-1},A_{k+1},\ldots,A_n\}.
$$

当这些矩阵 **互相 commute** 时，例如 Mamba 的 diagonal transition：

$$
A_iA_j=A_jA_i,
$$

PICASO-S 的 coefficient 可以写成 elementary symmetric polynomials：

$$
W_k
=
\frac1n
\sum_{m=0}^{n-1}
\frac{1}{\binom{n-1}{m}}
e_m(A_{-k}).
$$

这里 \(e_m\) 表示从其余 \(n-1\) 个 transition 中任选 m 个并求乘积后求和。

例如：

$$
e_0=I
$$

$$
e_1=A_1+A_2+\cdots
$$

$$
e_2=A_1A_2+A_1A_3+\cdots
$$

等等。

论文 Proposition 2 给出了利用 elementary symmetric polynomial 的 DP 算法；单个 coefficient 可通过 \(O(n^2)\) DP 求出，全部 coefficients 总复杂度为 \(O(n^3)\)。

最终：

$$
\boxed{
S_{PICASO-S}
=
\sum_{k=1}^{n} W_kH_k
}
$$

全程只有预计算 state/transition 上的算术运算。

---

# 6. PICASO-R：只平均 cyclic rotations

为了进一步降低组合开销，论文提出 PICASO-R。

不是平均全部 n! 个 permutation，而只考虑：

```text
A B C D
B C D A
C D A B
D A B C
```

一共 n 个 cyclic rotations。

因此它不是对整个 symmetric group 的完全 permutation invariance，而是对 cyclic permutation invariant。

对于第 k 个 context：

$$
W_k
=
\frac1n
\left[
I+
\sum_{m=1}^{n-1}
A_{[k+m]_n}\cdots A_{[k+1]_n}
\right].
$$

直观理解：

* 当 chunk k 位于最后时，它的 state 权重是 \(I\)；
* 当它后面还有一个 chunk 时，乘那个 chunk 的 A；
* 后面两个时，乘两个 transition；
* …
* 把它在所有 cyclic position 中受到的 future transitions 平均。

论文 Proposition 3 表明，如果各 \(A_i\) 可逆，可以通过 prefix/cumulative products 等方式在线性 \(O(n)\) 时间求出全部 PICASO-R coefficients。

### 对未来 GDN 复现尤其重要

PICASO-S 的上述高效公式明确要求：

```text
A_i commute
```

而 PICASO-R Proposition 3 的条件是：

```text
A_i invertible
```

**并不要求矩阵彼此 commute。**

所以如果以后把 PICASO 从 diagonal Mamba transition 扩展到 Gated DeltaNet 的 dense transition matrix，PICASO-R 在理论结构上反而比直接照搬 PICASO-S symmetric-polynomial DP 更自然。

---

# 7. 两个 chunk 时 PICASO 的含义最直观

只有 A、B 两个 chunk 时：

$$
S_{AB}=A_BH_A+H_B
$$

$$
S_{BA}=A_AH_B+H_A.
$$

对两种 permutation 平均：

$$
S_{PICASO}
=
\frac12(S_{AB}+S_{BA})
$$

于是：

$$
S_{PICASO}
=
\frac12(I+A_B)H_A
+
\frac12(I+A_A)H_B.
$$

所以 PICASO 不是：

```text
mean(H_A, H_B)
```

而是：

```text
transition-aware weighted state composition
```

这也是实现时最应该保留的核心语义。

---

# 8. Prepare / Compose / Query 的完整工作流

## Prepare：离线，每个 context 独立进行

对每个 context chunk \(u_i\)：

```text
1. 使用 zero initial state 独立 forward chunk_i
2. 保存所有 SSM layer 的 final recurrent state H_i
3. 同时计算并保存该 chunk 的累计 transition A_i
4. 原始 token 可保留给 retrieval，但在线生成不需要重新 forward
```

形成：

```text
PreparedChunk_i:
    layer 0:
        H_i^0
        A_i^0
    layer 1:
        H_i^1
        A_i^1
    ...
```

论文的方法是逐层处理这些 state/transition。

---

## Retrieve

收到 query 后：

```text
retrieve chunk ids:
    [i1, i2, ..., in]
```

retrieval 方法和 PICASO 本身无关。

论文实验使用普通文本 embedding retrieval；PICASO 只负责“检索完成后怎么组合 state”。

---

## Compose

对模型每个 SSM layer：

```text
states      = [H_i1, H_i2, ...]
transitions = [A_i1, A_i2, ...]

weights = PICASO_S(transitions)
# 或 PICASO_R(transitions)

composed_state = Σ weight_i @ state_i
```

注意：

```text
这里没有任何 context token model forward。
```

---

## Query Prefill / Generation

把：

```text
composed_state
```

安装为模型所有 SSM layer 的 initial state。

然后只 forward：

```text
query tokens
```

和后续 generation tokens。

这就是其 TTFT 优势的来源：context 长度不再进入在线模型 prefill 成本。论文报告 PICASO 平均约 5.4× faster than raw concatenation，并在实验中恢复了大部分 concatenation 带来的性能提升。

---

# 9. 多层模型中的一个重要近似

CASO Proposition 1 对一个独立 SSM layer 是精确的。

但真实 Mamba 是多层网络。

第 l 层 chunk B 的输入 hidden states，本身取决于：

```text
前面所有 layers
+
之前 context 的 states
```

而离线独立 Prepare B 时，这些 hidden states 是在：

```text
B 独立出现
```

的轨迹上产生的。

因此：

$$
(H_B^l,A_B^l)
$$

并不是“B 接在 A 后面”时该层真正会产生的 pair。

论文明确将 CASO 在 multilayer SSM 上视为对单层动力学关系的近似，而不是严格等价；它依靠实际模型中这种近似仍足够有效。

这点对于 Qwen3.8 更严重，因为 Qwen3.8 还是：

```text
GDN + Full Attention
```

hybrid model。

原始 PICASO 论文甚至明确把 attention-based hybrid model 以及 KV-cache composition 留作 future work，而不是声称已经解决。

---

# 10. Fine-tuning：不是复现第一阶段的必要部分

PICASO 首先是 training-free 方法，可以直接用 pretrained Mamba。

论文另外提出针对 composed states 做 fine-tuning：

* **BPTC**：Back-Propagation Through Composition，梯度经过 state composition；
* **BP2C**：把 composed state 当作停止梯度的输入，不反向穿过 composition。

其目的是让模型适应“人工组合出来的 state distribution”。

论文报告 fine-tuning 后可以接近 raw context concatenation 的质量，同时保留 composition 的速度优势。

对于当前 Qwen3.8 reuse 原型，不应该先做这一步。首先应验证 training-free composition。

---

# 11. 原论文中的主要 baseline

复现时有价值的几个 baseline：

### Raw Concat

```text
chunk1 tokens || chunk2 tokens || ... || query
```

全部正常 forward。

质量 reference。

### Soup

```text
S = mean(H_i)
```

完全不使用 transition。

最简单 state composition baseline。

### CASO

指定一个固定 context order：

```text
S = 0
for i:
    S = A_i S + H_i
```

有 transition awareness，但是 order-sensitive。

### PICASO-R

cyclic permutation averaged CASO。

重点优点是组合复杂度低。

### PICASO-S

full permutation averaged CASO。

理论上最完整的 permutation-invariant variant。

### PIConcat-R

真正按若干 permutation 重新 forward concatenated raw context，再平均它们得到的 state。

它可以作为高成本 oracle/reference，但已经失去了 reuse 的计算优势。

---

# 12. 对 Qwen3.8 复现者最重要的“原论文契约”

如果目的是把 PICASO 思想移植到 Qwen3.8 GDN，不要把 PICASO 理解成：

```text
prepare = 保存 final state
compose = 平均 final state
```

真正的 PICASO 数据契约是：

```text
prepare(chunk):
    对每一个 recurrent layer 保存

    H_chunk = F_chunk(0)
    P_chunk = chunk 对外部 initial state 的累计 transition

compose(chunks):
    根据 P_1 ... P_n 算 PICASO coefficient
    S = Σ W_i H_i

run:
    把 S 安装为 initial recurrent state
    只 forward query
```

对于 Mamba：

```text
P_chunk 是 diagonal
```

所以很便宜。

对于 Qwen3.8 GDN：

```text
P_chunk 将是 dense / per-head matrix
```

如何正确构造这个 P，就是移植 PICASO 时最核心的新工程问题。

原始 PICASO 本身没有解决 Full Attention KV cache；这部分必须由 hybrid-model reuse 方法另行处理。
