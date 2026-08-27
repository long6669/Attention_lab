# AttnLab MVP PRD

## 1. 项目名称

**AttnLab**

副标题：

> A visual playground for understanding and experimenting with attention mechanisms.

---

# 2. 项目背景

Attention 机制已经从传统 Multi-Head Attention 逐渐发展出 MQA、GQA、MLA、KDA、CSA、HCA 等大量不同结构。

现有学习资料主要存在几个问题：

1. 以静态公式和架构图为主，很难理解 Tensor 在不同阶段如何变化。
2. 很多资料只展示最终 Attention Heatmap，没有展示完整计算过程。
3. KV Cache、State、Compression 等推理过程比较抽象。
4. 不同 Attention 通常由不同代码实现，没有统一的观察方式。
5. 很难修改 Attention 结构并立即观察计算流程发生了什么变化。

AttnLab 希望提供一个：

> **可执行、可观察、可扩展的 Attention 学习与实验平台。**

第一阶段不追求真实大模型，也不追求性能，而是使用非常小的 Tensor，让用户能够真正看到 Attention 内部每一步发生了什么。

---

# 3. MVP 目标

第一版只实现一个完整闭环：

```text
输入约 10 个 token

↓

生成 Toy Embedding

↓

执行 Multi-Head Attention

↓

记录每一步计算过程

↓

自动生成 Attention Graph

↓

前端动态播放计算过程

↓

查看 Tensor Shape / Tensor Value

↓

查看 Attention Matrix

↓

查看 KV Cache

↓

执行一个 Decode Step

↓

观察 KV Cache 增长
```

第一版的核心目标不是支持大量 Attention，而是验证整个框架：

> **Attention Code → Graph → Runtime → Trace → Visualization**

能够正常工作。

---

# 4. MVP 非目标

第一版明确不实现：

- 真实大语言模型
- HuggingFace 模型加载
- CUDA
- Triton
- FlashAttention
- 模型训练
- Autograd
- KDA
- CSA
- HCA
- MLA
- Graph 拖拽编辑
- 自定义 Attention 在线编程
- LLM 自动生成 Attention
- LLM 自动解释 Graph
- 多用户系统
- 用户登录
- 数据库存储
- 云端部署优化
- 超大 Tensor 可视化
- 高性能计算

这些属于后续版本。

---

# 5. 用户故事

## 5.1 学习 Attention

作为用户，我输入：

```text
I love learning how attention works today
```

系统将文本拆成 token，并生成简单 embedding。

我点击：

```text
Run
```

之后看到 Attention 的完整 Graph。

我可以点击：

```text
Next
```

依次观察：

```text
Input
↓
Q Projection
↓
K Projection
↓
V Projection
↓
Split Heads
↓
QKᵀ
↓
Scale
↓
Causal Mask
↓
Softmax
↓
Attention × V
↓
Merge Heads
↓
Output
```

---

## 5.2 查看 Tensor

当运行到某一个节点时，我可以点击节点。

例如：

```text
Q Projection
```

右侧显示：

```text
Tensor Name
Q

Shape
[2, 10, 4]

dtype
float32
```

并能够查看部分实际数值。

---

## 5.3 查看 Attention Matrix

当运行到：

```text
QKᵀ
Scale
Mask
Softmax
```

时，下方显示对应矩阵。

用户可以看到：

```text
Raw Score
↓
Scaled Score
↓
Masked Score
↓
Attention Probability
```

是如何一步一步变化的。

---

## 5.4 查看 KV Cache

完成 Prefill 后展示：

```text
K Cache
shape = [1, 2, 10, 4]

V Cache
shape = [1, 2, 10, 4]

Total Elements = 160
Memory = 640 Bytes FP32
```

---

## 5.5 Decode 一个 Token

用户点击：

```text
Decode One Token
```

系统新增一个 toy token。

例如：

```text
token_11
```

然后展示：

```text
K Cache

10 tokens
↓
11 tokens
```

以及：

```text
640 B
↓
704 B
```

让用户直观理解 KV Cache 为什么会随 Sequence Length 增长。

---

# 6. 产品页面

MVP 只需要一个页面。

页面结构：

```text
┌──────────────────────────────────────────────┐
│ AttnLab                                      │
│                                              │
│ Input Tokens                  [Run]          │
├──────────────────────────────────────────────┤
│                                              │
│                Graph View                    │
│                                              │
│     Input                                    │
│       │                                      │
│     Q K V                                    │
│       │                                      │
│     Attention                                │
│       │                                      │
│     Output                                   │
│                                              │
├───────────────────────┬──────────────────────┤
│ Attention Matrix      │ Tensor Inspector     │
│                       │                      │
│                       │ Name                 │
│                       │ Shape                │
│                       │ Values               │
│                       │                      │
├───────────────────────┴──────────────────────┤
│ KV Cache                                     │
│                                              │
│ K Cache                                      │
│ V Cache                                      │
│ Memory Usage                                 │
├──────────────────────────────────────────────┤
│ Previous     Play / Pause      Next          │
│                                              │
│ Decode One Token                             │
└──────────────────────────────────────────────┘
```

---

# 7. 默认配置

为了保证展示清晰，MVP 默认使用：

```text
batch_size = 1

seq_len <= 10

d_model = 8

num_heads = 2

head_dim = 4

dtype = float32
```

不允许用户在第一版修改模型维度。

后续再开放参数配置。

---

# 8. Token 与 Embedding

MVP 不使用真实 tokenizer。

使用最简单方案：

```python
text.split()
```

例如：

```text
I love attention
```

得到：

```text
["I", "love", "attention"]
```

最多允许 10 个 token。

Embedding 使用 deterministic random。

要求：

```text
相同 token
+
相同 seed

→

相同 embedding
```

默认：

```python
seed = 42
```

这样方便调试和重复实验。

---

# 9. 核心架构

项目采用：

```text
Attention Definition
        ↓
Attention Graph
        ↓
Runtime
        ↓
Trace Recorder
        ↓
API
        ↓
Frontend Visualization
```

---

# 10. Attention Graph

虽然 MVP 只实现 MHA，但不能把 MHA 写死在前端。

后端需要把 Attention 表达成 Graph。

Graph 包含：

```text
Node
Edge
Tensor
```

---

## 10.1 Node

建议结构：

```python
@dataclass
class Node:
    id: str
    op: str
    label: str

    inputs: list[str]
    outputs: list[str]

    attrs: dict
```

示例：

```python
Node(
    id="q_proj",
    op="linear",
    label="Q Projection",
    inputs=["x"],
    outputs=["q"],
    attrs={}
)
```

---

## 10.2 TensorSpec

```python
@dataclass
class TensorSpec:
    id: str
    name: str
    shape: tuple
    dtype: str
```

例如：

```python
TensorSpec(
    id="tensor_q",
    name="Q",
    shape=(2, 10, 4),
    dtype="float32"
)
```

---

## 10.3 Graph

```python
@dataclass
class Graph:
    nodes: list[Node]
    edges: list
    tensors: dict
```

---

# 11. MVP Primitive

第一版只实现以下 primitive：

```text
Input

Embedding

Linear

SplitHeads

Transpose

MatMul

Scale

CausalMask

Softmax

MergeHeads

CacheAppend

Output
```

不要为了未来需求提前实现大量 primitive。

---

# 12. MHA 实现

MHA 不能直接作为黑盒函数。

它需要使用 Graph primitive 构建。

伪代码：

```python
def build_mha(g, x):

    q = g.linear(x, "q_proj")
    k = g.linear(x, "k_proj")
    v = g.linear(x, "v_proj")

    q = g.split_heads(q)
    k = g.split_heads(k)
    v = g.split_heads(v)

    k_cache = g.cache_append(k, "k_cache")
    v_cache = g.cache_append(v, "v_cache")

    scores = g.matmul(
        q,
        g.transpose(k_cache)
    )

    scores = g.scale(scores)

    scores = g.causal_mask(scores)

    probs = g.softmax(scores)

    output = g.matmul(
        probs,
        v_cache
    )

    output = g.merge_heads(output)

    return output
```

---

# 13. Runtime

第一版使用：

```text
NumPy
```

实现：

```python
class NumPyRuntime:
    ...
```

负责真正执行：

```text
Linear
MatMul
Softmax
Reshape
Transpose
Mask
```

前端不执行任何 Attention 数学。

---

# 14. Trace Recorder

每执行一个 Graph Node，需要记录一个 Trace Event。

例如：

```python
@dataclass
class TraceEvent:
    step: int

    node_id: str

    op: str

    inputs: list[str]

    outputs: list[str]

    title: str
```

执行：

```python
q = runtime.linear(x, Wq)
```

同时：

```python
recorder.record(...)
```

---

# 15. Tensor Store

由于 MVP Tensor 很小，可以直接保存完整数据。

结构：

```json
{
    "tensor_q": {
        "name": "Q",
        "shape": [2, 10, 4],
        "dtype": "float32",
        "values": []
    }
}
```

未来 Tensor 较大时再增加：

```text
summary mode
sample mode
metadata mode
```

MVP 暂时不考虑。

---

# 16. 后端 API

使用：

```text
FastAPI
```

---

## 16.1 Run Attention

```http
POST /api/run
```

Request：

```json
{
    "text": "I love learning attention"
}
```

Response：

```json
{
    "tokens": [],

    "graph": {
        "nodes": [],
        "edges": []
    },

    "trace": [],

    "tensors": {},

    "memory": {
        "k_cache": {},
        "v_cache": {},
        "total_elements": 0,
        "total_bytes": 0
    }
}
```

---

## 16.2 Decode One Token

```http
POST /api/decode
```

MVP 可以暂时不维护复杂 session。

前端直接将当前 tokens 再传回来。

例如：

```json
{
    "tokens": [
        "I",
        "love",
        "attention"
    ]
}
```

后端增加：

```text
token_4
```

重新执行并返回新的结果。

第一版优先简单实现。

后续再增加真正的 Stateful Runtime。

---

# 17. Frontend

使用：

```text
React
TypeScript
Vite
```

---

# 18. Graph View

推荐：

```text
React Flow
+
ELK.js
```

Graph View 不允许针对 MHA 写固定坐标。

后端只返回：

```text
nodes
edges
```

前端通过 ELK 自动布局。

流程：

```text
Graph JSON
↓
Graph Adapter
↓
ELK Layout
↓
React Flow
```

---

# 19. Graph Node

第一版只做两类：

## DefaultNode

所有普通 operator 共用。

显示：

```text
Operator Name

Input Shape
↓

Output Shape
```

例如：

```text
┌──────────────┐
│ Softmax      │
│              │
│ [2,10,10]   │
│      ↓       │
│ [2,10,10]   │
└──────────────┘
```

---

## CacheNode

Cache 使用特殊样式。

例如：

```text
┌──────────────┐
│ KV Cache     │
│              │
│ 10 Tokens    │
│ 640 Bytes    │
└──────────────┘
```

其他特殊 Node Renderer 后续增加。

---

# 20. Generic Fallback

如果前端遇到未知：

```text
op
```

必须使用：

```text
GenericNode
```

不能导致页面崩溃。

原则：

> 所有合法 Graph 都必须能够被展示。

即使没有专门动画。

---

# 21. Graph Timeline

Graph 下方提供：

```text
Previous

Play / Pause

Next
```

状态：

```typescript
currentStep
```

当前 Trace Event 对应 Node：

```text
高亮
```

相关 Edge：

```text
高亮
```

其他 Node：

```text
降低透明度
```

---

# 22. Tensor Inspector

点击 Node 后展示：

```text
Operator
Input Tensor
Output Tensor

Shape

dtype

Values
```

MVP 直接显示完整小 Tensor。

最大 Tensor：

```text
10 × 10
```

所以不会造成性能问题。

---

# 23. Attention Matrix

对于以下步骤：

```text
QK MatMul

Scale

Mask

Softmax
```

显示矩阵。

矩阵维度：

```text
10 × 10
```

支持 Head：

```text
Head 0

Head 1
```

切换。

第一版 Heatmap 可以使用：

```text
HTML Grid
```

或者：

```text
SVG
```

不需要复杂图表库。

---

# 24. KV Cache View

展示：

```text
Current Token Count

K Shape

V Shape

K Elements

V Elements

Total Elements

Memory Bytes
```

例如：

```text
Tokens

10

K

[1,2,10,4]

80 values

V

[1,2,10,4]

80 values

Total

160 values

640 Bytes
```

---

# 25. Decode 动画

点击：

```text
Decode One Token
```

增加：

```text
token_11
```

KV Cache View 更新：

```text
Tokens

10 → 11
```

Memory：

```text
640 B → 704 B
```

新增部分需要视觉高亮。

---

# 26. 前端状态

第一版不引入 Redux。

使用：

```text
React useState
```

如果状态开始复杂，再使用：

```text
Zustand
```

---

# 27. 项目结构

建议：

```text
attnlab/

backend/

    app/

        main.py

        ir/
            graph.py
            node.py
            tensor.py

        ops/
            primitives.py

        runtime/
            numpy_runtime.py

        tracing/
            recorder.py

        architectures/
            mha.py

        api/
            attention.py


frontend/

    src/

        components/

            GraphView/
            TensorInspector/
            AttentionMatrix/
            KVCacheView/
            Timeline/

        graph/
            adapter.ts
            layout.ts

        api/
            attention.ts

        types/
            graph.ts
            trace.ts
            tensor.ts

        App.tsx
```

---

# 28. UI 原则

整体风格：

```text
简洁
技术感
低干扰
Debug Tool 风格
```

不要做：

```text
炫酷 3D
复杂粒子动画
大量渐变
复杂 Dashboard
```

重点应该是：

```text
图清楚

Tensor 清楚

数据流清楚

当前 Step 清楚
```

---

# 29. 核心交互

必须实现：

```text
输入 Text

Run

Previous

Next

Play

Pause

点击 Node

切换 Head

Decode One Token
```

第一版没有其他交互要求。

---

# 30. 错误处理

以下情况需要提示：

### 空输入

```text
Please enter some tokens.
```

### 超过 10 Tokens

自动截断或提示：

```text
MVP supports up to 10 tokens.
```

推荐直接截断并提示。

### Backend Error

显示：

```text
Failed to run attention.
```

不能让整个页面崩溃。

---

# 31. 测试要求

Backend 至少测试：

```text
Embedding Shape

Q Shape

K Shape

V Shape

Attention Score Shape

Softmax Row Sum ≈ 1

Causal Mask

Output Shape

KV Cache Shape

KV Cache Memory Calculation
```

例如：

```python
assert np.allclose(
    attention_probs.sum(axis=-1),
    1
)
```

---

# 32. MVP 验收标准

当用户输入：

```text
I love learning how attention works today
```

点击：

```text
Run
```

系统必须：

### 1

正确拆分 tokens。

### 2

生成 toy embedding。

### 3

真实执行 MHA。

### 4

生成动态 Graph。

### 5

Graph 不是写死坐标。

### 6

可以 Next / Previous。

### 7

当前 Node 正确高亮。

### 8

可以点击 Node 查看 Tensor。

### 9

能够看到 QKᵀ。

### 10

能够看到 Mask 前后变化。

### 11

能够看到 Softmax Matrix。

### 12

能够查看 KV Cache。

### 13

能够看到 KV Cache 字节数。

### 14

点击 Decode One Token 后：

```text
10 tokens
→
11 tokens
```

### 15

KV Cache 同步增长。

做到以上功能，即认为 MVP 完成。

---

# 33. 第一阶段开发顺序

请严格按照下面顺序实现，不要一次性构建整个系统。

## Step 1

实现：

```text
TensorSpec
Node
Graph
```

---

## Step 2

实现：

```text
NumPyRuntime
```

---

## Step 3

实现：

```text
MHA
```

确保能够独立通过 Python 测试。

---

## Step 4

实现：

```text
TraceRecorder
```

确保：

```text
run()
```

能够返回：

```text
Graph
Trace
Tensor
Memory
```

---

## Step 5

实现 FastAPI：

```text
POST /api/run
```

---

## Step 6

初始化 React 前端。

---

## Step 7

实现：

```text
Graph JSON
↓
ELK
↓
React Flow
```

---

## Step 8

实现 Timeline。

---

## Step 9

实现 Tensor Inspector。

---

## Step 10

实现 Attention Matrix。

---

## Step 11

实现 KV Cache View。

---

## Step 12

实现 Decode One Token。

---

# 34. 后续扩展设计

MVP 完成以后，再逐步加入：

```text
V0.2

MQA
GQA
```

验证：

```text
num_q_heads
和
num_kv_heads
```

的差异。

---

```text
V0.3

RoPE
```

---

```text
V0.4

MLA
```

加入：

```text
LowRankCompression

Latent Cache
```

---

```text
V0.5

KDA
```

加入：

```text
State

Scan

Decay

Erase

Write
```

---

```text
V0.6

CSA / HCA
```

加入：

```text
SequenceCompression

Indexer

TopK

Routing
```

---

```text
V0.7

Concept Graph
```

加入 LLM：

```text
Raw Graph
↓
LLM Semantic Analysis
↓
Concept Graph
```

---

```text
V1.0

Custom Attention Playground
```

允许用户：

```text
修改 Attention Graph

运行

比较

查看 Tensor

查看 Memory
```

---

# 35. LLM 的未来边界

MVP 不接 LLM。

未来 LLM 主要用于：

```text
解释 Graph

生成 Concept Graph

解释陌生 Operator

分析自定义 Attention

辅助生成 Attention IR
```

LLM 不负责：

```text
Attention 数学计算

Tensor 数据

Graph Edge

Graph Layout 坐标

Runtime
```

原则：

> Runtime 决定真实发生了什么。

> Graph Layout Engine 决定节点放在哪里。

> LLM 帮助用户理解发生了什么。

---

# 36. 最终 MVP 定义

第一版最终只需要把这一条链路做好：

```text
10 Tokens
    ↓
Toy Embedding
    ↓
MHA
    ↓
Attention Graph
    ↓
NumPy Runtime
    ↓
Execution Trace
    ↓
FastAPI
    ↓
React
    ↓
ELK Dynamic Graph
    ↓
Tensor Inspector
    ↓
Attention Matrix
    ↓
KV Cache
    ↓
Decode One Token
```

不要在 MVP 阶段继续扩大需求。

这个版本完成以后，再以现有 Graph / Runtime / Trace 架构为基础扩展 MLA、KDA、CSA、HCA 和自定义 Attention。

核心工程原则：

> **Architecture should be extensible, implementation should stay minimal.**