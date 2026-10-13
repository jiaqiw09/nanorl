# Megatron 原生训练后端

训练入口默认：`--actor-backend megatron --actor-dtype bfloat16`。
当前实现单训练 rank、TP=PP=DP=1；功能已接入，GPU 运行与数值验证后置。

```bash
bash scripts/run_demo.sh /models/qwen \
  --actor-backend megatron --actor-dtype bfloat16 \
  --weight-sync nccl --rollout-dtype bfloat16 \
  --rollout-concurrency 4 --reward exact-integer
```

## 模型和优化器的建立顺序

1. Bridge 用 BF16 provider 建立 Megatron 模型，加载 HF 初始权重。
2. 保留底层模型引用，供 HF 导出与 NCCL 权重转换读取。
3. 用 MCore `Float16Module` 处理 BF16 前向输入/输出边界。
4. 用 MCore `DistributedDataParallel` 分配连续梯度缓冲，设置 `grad_reduce_in_fp32=True`。
5. 用 `get_megatron_optimizer(OptimizerConfig(...))` 创建原生 Adam 优化器。

训练 forward 使用 DDP 包装后的模型；转换使用同一批参数的底层模型，不建立第二份训练模型。
HF 权重必须在 optimizer 创建前加载，否则 FP32 主参数可能保留加载前的旧值。

| 数据 | 默认 dtype / 所有者 |
|---|---|
| 前向/反向模型参数 | BF16，Megatron 模型 |
| 累积梯度 `main_grad` | FP32，Megatron DDP |
| optimizer 主参数 | FP32，Megatron 混合精度 optimizer |
| Adam moments | FP32，原生 Adam 实现 |
| GRPO logprob 与 loss | FP32，actor loss 计算 |
| 推送给 SGLang 的参数 | 当前模型参数，转换为 `--rollout-dtype` |

这里关闭 `use_distributed_optimizer`，因为只有一个训练 rank，不需要分片 optimizer 状态。
关闭梯度通信重叠和参数拉取重叠；保留显式的累积、同步、更新顺序。
也关闭 gradient accumulation fusion，DDP hook 将普通反向梯度加入 `main_grad`，避免强依赖额外 fused extension。

## 一轮更新

```text
DDP.zero_grad_buffer() + optimizer.zero_grad()
    ↓
microbatch forward → 每条回答 token mean → loss 之和 / 整轮回答数
    ↓
optimizer.scale_loss(loss).backward() → 累积 FP32 main_grad
    ↓
main_grad 有限值检查 → finalize_model_grads()
    ↓
Megatron optimizer.step()
    ├─ gradient clipping
    ├─ FP32 主参数/Adam 状态更新
    └─ FP32 主参数回写 BF16 模型
    ↓
成功才递增 policy_version → 磁盘或 NCCL 同步
```

不要对 DDP 模型的 `.grad` 使用 PyTorch 裁剪函数：Megatron hook 可以将 `.grad` 清空，
真实累积结果在 `main_grad`。原生 optimizer 完成裁剪并返回 grad_norm。
也不能只清 optimizer 梯度而不清 DDP buffer，否则不同训练轮之间会残留梯度。

BF16 不启用动态 loss scaling；仍通过原生 `scale_loss` 接口反向。
optimizer 返回 unsuccessful 时记录 `optimizer_skipped=true`，保持策略版本、不推送新权重。
全零优势则在 backward 前直接跳过本轮。

`--train-microbatch-size` 控制每次 forward/backward 的回答数，默认 1。多回答时右侧 padding，
所有 microbatch 累积后只做一次 optimizer step。PP=1 不调用流水线调度器；
PP schedule、多 rank 分布式优化器仍不在当前范围。
学习率保持固定，不配置 warmup/decay scheduler。

训练 checkpoint 同时保存底层模型与原生 optimizer 的完整 state_dict，包含 FP32 主参数和
Adam 状态。恢复先加载模型，再加载 optimizer；不能再用 BF16 模型刷新 FP32 主参数，
否则会丢掉断点中保留的精度。保存发生在整轮完成之后，梯度缓冲不需要跨断点保留。
恢复入口和目录见 [训练状态说明](training.md#checkpoint-与-resume)。

## 对照路径和转换工具

```bash
# 原来的 FP32 Megatron Core 模型 + PyTorch AdamW
bash scripts/run_demo.sh /models/qwen --actor-backend torch --actor-dtype float32

# 原生 Megatron DDP/optimizer，也可用 FP32 模型
bash scripts/run_demo.sh /models/qwen --actor-backend megatron --actor-dtype float32
```

`roundtrip` 和既有 `check_actor` 命令保持独立的 FP32 基线路径，其结果不能作为原生 BF16 路径已验证的证据。
BF16 主参数上的小更新不一定立刻改变 BF16 模型值；策略版本表示成功执行 optimizer step，
不证明每次都有可观察的推理变化。

源码入口：`nanorl/megatron_backend.py`、`nanorl/actor.py`、`nanorl/conversion.py`。
实现对照本地 slime 的 `megatron_utils/model.py` 及
[MCore optimizer](https://github.com/NVIDIA/Megatron-LM/blob/1dcf0dafa884ad52ffb243625717a3471643e087/megatron/core/optimizer/__init__.py)。
