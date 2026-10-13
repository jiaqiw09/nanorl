# HF ↔ Megatron 权重转换与往返验收

实现：`nanorl/conversion.py` 中 `QwenConversion` 复用 NVIDIA Megatron Bridge 的实际转换。
actor 加载和每轮导出都走该模块；校验器中的 `hf_views()` 是独立的检查实现，不参与正式导出。
当前范围为单 rank、TP=PP=1、dense Qwen2.5，不处理 TP gather、PP、MoE 或量化。

## 名称与布局

设 hidden size 为 H，MLP intermediate size 为 F，query heads 为 A，KV groups 为 G，
head dimension 为 D=H/A，每个组含 Q=A/G 个 query heads。

| HF 参数 | Megatron 参数 | 布局约定 |
|---|---|---|
| `model.embed_tokens.weight` | `embedding.word_embeddings.weight` | `[vocab,H]`；Megatron 可能在词表尾部补齐 |
| `lm_head.weight` | `output_layer.weight` | tied 时共享 embedding；HF 可以省略重复的 head |
| `model.norm.weight` | `decoder.final_layernorm.weight` | `[H]` |
| `layers.i.input_layernorm.weight` | `decoder.layers.i.self_attention.linear_qkv.layer_norm_weight` | TE 将 RMSNorm 融入线性层 |
| `layers.i.post_attention_layernorm.weight` | `decoder.layers.i.mlp.linear_fc1.layer_norm_weight` | `[H]` |
| `layers.i.self_attn.{q,k,v}_proj.weight` | `decoder.layers.i.self_attention.linear_qkv.weight` | `[G,Q+2,D,H]`，每组依次为 Q 个 query、一个 key、一个 value |
| `layers.i.self_attn.{q,k,v}_proj.bias` | `decoder.layers.i.self_attention.linear_qkv.bias` | `[G,Q+2,D]`，与 weight 的组顺序一致 |
| `layers.i.self_attn.o_proj.weight` | `decoder.layers.i.self_attention.linear_proj.weight` | `[H,H]` |
| `layers.i.mlp.{gate,up}_proj.weight` | `decoder.layers.i.mlp.linear_fc1.weight` | `[2F,H]`，前 F 行 gate、后 F 行 up |
| `layers.i.mlp.down_proj.weight` | `decoder.layers.i.mlp.linear_fc2.weight` | `[H,F]` |

表中 HF 的 `layers.i` 前缀实际为 `model.layers.i`。
GQA 的 fused QKV 不是把整个 Q、整个 K、整个 V 简单拼起来；这是最容易遗漏的转换细节。
词表 padding 只在检查 embedding/head 时裁掉，其他 tensor 必须 shape 完全相同。
词表 padding 对前向/logprob 的影响应在后续 actor 数值验收中另外检查。

## 真正的一卡往返

在已经包含 CUDA / Megatron / Bridge 的 Linux 环境执行：

```bash
cd /absolute/path/to/nanorl
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29501 python3 -m nanorl.roundtrip \
  --model-path /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --output-dir runs/roundtrip-001
```

输出目录必须不存在，防止覆盖旧结果。产物包括 `exported_hf/` 和 `roundtrip.json`。
任一阶段失败时命令返回非零，并保留失败报告；不会因为没有 CUDA 而输出成功。

检查按顺序执行：

1. 运行环境、Bridge API、checkpoint shard/index、tensor 名称/shape/dtype、tokenizer 预检查。
2. 实际调用 Bridge 将 HF 权重加载进 GPU 上的 Megatron 模型。
3. 独立检查 **Megatron 中间态** 的每个参数与原始 HF 的对应关系，检查 QKV bias、GQA、
   gate/up 和 norm；检查模型确实启用 tied embedding，共享的 output weight 不能是另一份独立参数。
4. Bridge 导出 HF，重新解析全部 shard，检查 config、tokenizer/chat template、权重。
5. 逐 tensor 比较原始 HF 和导出 HF，记录源/目标 dtype、元素数和 max absolute error。

本测试不执行 optimizer step，不调用 SGLang，避免将训练变化或推理误差混入转换检查。
单纯 reshape/split/concat 应保持数值，所以权重比较使用 **atol=rtol=0**，不随意放宽容差。
原始 FP16/BF16 → actor FP32 → 导出 FP32 是允许的无损扩宽；导出也允许保留原始 dtype，
但必须逐值相等。拒绝意外降精度、NaN/Inf、缺失或额外键、不匹配的索引、不同值的 tied head。
如果往返失败，报告给出第一个失败 tensor；已经完成阶段的逐 tensor 指标仍会写入报告。

只验证 HF→Megatron→HF 两端可能漏掉“正反转换共享同一个错误”的情况，因此第 3 步不能省略。
CPU 测试用可辨认的行号检查 QKV 顺序，并包含缺失 shard、错误索引、损坏参数、shape、
tied embeddings 和 dtype 变化测试；这些仍不能代替真实 Bridge 往返。

参考：[NVIDIA Qwen2 Bridge 映射](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen2_bridge.py)、
[AutoBridge 加载/导出 API](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/auto_bridge.py)。
文档中的上游链接用于解释；运行契约通过实际安装版本的 API 检查和实机测试确认。
