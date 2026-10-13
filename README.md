# nanorl：两张卡上的最小 GRPO 闭环

**状态：功能开发中，实机验证后置。** 已实现环境/启动、Bridge 转换、样本与版本契约、Megatron 原生 BF16 更新、有界并发 rollout、可选择 reward、磁盘/NCCL 同步，以及训练 microbatch、checkpoint/resume、可恢复采样器和持久日志。功能已接入训练循环，尚未实机运行；按要求不添加评估。

这是一个用来读懂 RL 训练数据流的同步式 demo。它从 `slime` 的 SGLang/Megatron 权重同步路径和 `verl` 的 GRPO 分组优势实现提炼出必要步骤，代码独立放在本目录，不修改两个框架。默认使用本地 `Qwen/Qwen2.5-0.5B-Instruct` 的 Hugging Face 权重。

## 目录

```text
nanorl/
├── README.md                说明、运行和边界
├── requirements.txt         直接依赖清单
├── docs/environment.md      固定镜像 digest、环境检查、启动约束
├── docs/conversion.md       QKV/GQA 等映射说明与往返验收
├── docs/training.md         样本契约、GRPO 更新与终止语义
├── docs/weight_sync.md      磁盘/NCCL 权重同步流程与使用方法
├── docs/megatron.md         BF16、DDP main_grad、原生 optimizer 的完整更新流程
├── examples/math.jsonl      prompt 和正确整数答案
├── scripts/run_demo.sh      GPU 1 推理、GPU 0 训练的启动脚本
├── scripts/run_container.sh 使用固定 digest 的完整 CUDA 镜像
├── nanorl/
│   ├── config.py            CLI 配置与输入检查
│   ├── environment.py       模型/依赖 API/CUDA 预检查与环境记录
│   ├── launch.py            非共卡分配、服务身份和进程生命周期
│   ├── conversion.py        Bridge 转换封装、checkpoint 与布局校验
│   ├── roundtrip.py         真正的 HF→Megatron→HF 一卡验收
│   ├── check_actor.py       后期使用的受控 actor 更新验收入口
│   ├── rollout.py           调用 SGLang /generate，记录 token/旧 logprob
│   ├── reward.py            内置规则奖励与 module:function 加载
│   ├── grpo.py              组内优势和 PPO 式 clip surrogate
│   ├── actor.py             Megatron Core actor 建立、前向及更新
│   ├── batching.py          右侧 padding、response mask、批量 logprob
│   ├── megatron_backend.py  原生 DDP、FP32 梯度缓冲与混合精度 optimizer
│   ├── sampler.py           epoch/shuffle、数据指纹及可恢复位置
│   ├── checkpoint.py        模型/optimizer/RNG/sampler 原子存档与恢复
│   ├── run_log.py           配置快照、按 session 追加持久 JSONL
│   ├── sync.py              Megatron Bridge 导出 HF、SGLang 磁盘热更新
│   ├── sync_nccl.py          HF tensor 流、独立 NCCL 通信组、跨卡广播
│   └── train.py             按版本串起完整训练循环
└── tests/                   CPU 可跑的分组、奖励和接口检查
```

## 一轮在做什么

```text
HF 初始权重 → Megatron actor (GPU 0)
                      │ Megatron Bridge 转换成 HF 名称与布局
                      ├── disk：完整 HF 文件 → 热加载
                      └── nccl：流式 tensor → 跨卡广播
                                      │
                               SGLang (GPU 1)
                                      │ 一个 prompt 采样 G 个回答
                                      ▼
                             token + 旧 logprob + 文本
                                      │
                         exact-integer reward，组内标准化
                                      ▼
                         Megatron actor 上算新 logprob
                                      │ clipped GRPO loss
                                      ▼
                              optimizer.step()，版本 +1
```

每个 prompt 的 G 个奖励单独计算 `Aᵢ = (rᵢ - 组均值) / (组标准差 + 1e-6)`；组内奖励全部相同时优势为零。对回答 token 使用 `min(ratio×A, clip(ratio, 1-ε, 1+ε)×A)`，其中 `ratio = exp(new_logprob - old_logprob)`。每个回答先对 token 取均值，再对回答取均值。只训练回答 token，没有 critic、KL 惩罚或 reference model。采样固定为 `temperature=1, top_p=1`，让 SGLang 回传的旧 logprob 与 Megatron 的原始策略 logprob 对应。

## 运行

直接跨卡推权重（无需每轮落 HF 文件）：

```bash
TRAIN_GPU=0 ROLLOUT_GPU=1 bash scripts/run_demo.sh /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --actor-backend megatron --actor-dtype bfloat16 \
  --weight-sync nccl --rollout-dtype bfloat16 --sync-bucket-mb 64 \
  --rollout-concurrency 4 --reward exact-integer --train-microbatch-size 2
```

完整机制见 [权重同步说明](docs/weight_sync.md)。默认仍为 `--weight-sync disk`。
actor 默认是 Megatron 原生 BF16 路径；原来的基线用 `--actor-backend torch --actor-dtype float32` 选择。
详细过程见 [Megatron 后端](docs/megatron.md)，奖励扩展见 [训练契约](docs/training.md)。

需要 Linux、Python 3.10+、CUDA、至少两张可用 GPU，以及完整的 SGLang / Megatron Bridge / Core / Transformer Engine 环境。[环境文档](docs/environment.md) 说明如何固定完整镜像 digest；[单卡权重往返](docs/conversion.md) 留到后期验证阶段执行。`requirements.txt` 仅是依赖清单，不是已验证的 lockfile。模型目录需包含完整 HF safetensors 权重和 tokenizer，并对两个进程都可见。

在本目录运行：

```bash
cd nanorl
TRAIN_GPU=0 ROLLOUT_GPU=1 bash scripts/run_demo.sh /absolute/path/to/Qwen2.5-0.5B-Instruct
```

脚本可从任意目录调用，先检查两张物理卡、依赖和模型，然后启动 SGLang；健康、模型路径和本次实例身份都符合才启动训练。运行产物保存在 `runs/<时间戳>-<PID>/`：`metrics.jsonl` 为训练事件/指标，`sessions/` 为配置快照，`checkpoints/` 为可恢复训练状态，`weights/<session>/weight_vNNNNNN` 为磁盘同步的 HF 权重，`sglang.log` 为服务日志；环境记录为 `environment-{train,rollout}.json`。可用 `RUN_DIR`、`PORT`、`MASTER_PORT`、`TRAIN_GPU`、`ROLLOUT_GPU`、`STARTUP_TIMEOUT` 调整。

### Microbatch 与恢复训练

```bash
RUN_DIR="$PWD/runs/demo" bash scripts/run_demo.sh /models/qwen \
  --steps 20 --train-microbatch-size 2 --checkpoint-every 2 --seed 42

# 恢复后训练到累计 40 步；保留原训练参数。可使用同一或新的 RUN_DIR。
RUN_DIR="$PWD/runs/demo" bash scripts/run_demo.sh /models/qwen \
  --steps 40 --train-microbatch-size 2 --checkpoint-every 2 --seed 42 \
  --resume "$PWD/runs/demo/checkpoints/latest.json"
```

默认 microbatch_size=1、checkpoint_every=1、seed=42，每个 epoch shuffle（`--no-shuffle` 关闭）。
断点同时包含模型、optimizer、策略版本、RNG 与数据位置；恢复后先同步 SGLang 再采样。
`--steps` 是累计目标，失败半轮从上一个完整断点重做；HF 发布目录不能用于 `--resume`。
launcher 开启 SGLang deterministic inference 使请求级 seed 生效，内核支持和实际结果仍待实机验证。
详见 [训练、断点与日志说明](docs/training.md)。

单独执行真实转换检查（仅需一张 GPU，输出目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m nanorl.roundtrip \
  --model-path /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --output-dir runs/roundtrip-001
```

检查 Megatron 中间参数、导出 HF 全部 tensor、配置和 tokenizer，并写入 `roundtrip.json`；失败返回非零。

后期具备依赖时可运行既有测试（本轮未新增或执行）：

```bash
python -m unittest discover -s tests -v
```

## 约束

- 这是单机、单训练 rank、TP=PP=DP=1 的 Megatron Core 演示。默认使用 MCore DDP 和原生 optimizer，另有 PyTorch AdamW 基线；没有 optimizer 分片、PP 流水线或多节点支持。
- `disk` 模式导出 HF 文件再热加载；`nccl` 模式流式转换 HF tensor 并通过独立通信组广播，两种方式均暂停生成、清缓存、检查版本后恢复。当前只支持单机训练/推理各一张卡。
- 原生后端使用 BF16 模型、FP32 累积梯度、FP32 主参数及 Adam 状态；显存仍需容纳完整单卡训练状态。支持多回答 padding microbatch，数学目标不变，数值等价性尚待验证。
- 对全错或全对的组，GRPO 优势为零；整批全零时跳过 optimizer 和同步，保持策略版本。提供 `examples/math_mixed.jsonl`，但实际有效组仍取决于采样结果。
- 磁盘模式按 session 创建版本目录；NCCL 模式不创建 HF 文件，但两种模式都保存训练断点。断点及发布版本暂时全部保留，可调整 checkpoint 间隔；完整 Adam 状态需要额外磁盘和恢复时的 CPU 内存。
- 权重更新失败或版本号不符时，中止训练并保持 SGLang 暂停。恢复需从已知有效的 checkpoint 重启服务；当前没有自动回滚。

## 尚待完成的两卡验收

1. 在固定 digest 的候选镜像上实际通过环境预检查，保存对应报告；当前没有可宣称已通过的版本组合。
2. 执行已新增的 `nanorl.roundtrip`，核对参数名称、形状、dtype 和数值，特别是 QKV/GQA 排列、gate/up 拆分与 tied embeddings。
3. 对固定 prompt 与固定 continuation 比较 Megatron、导出 HF 模型和 SGLang 的逐 token logprob，并明确允许的 dtype/内核数值误差。
4. 用非零优势的受控 batch 更新一次，确认有限梯度与非零参数变化；同步后确认 SGLang 的固定 continuation logprob 随新策略变化。版本号本身不证明权重数值正确。

参考：[`slime` 的磁盘同步实现](https://github.com/THUDM/slime/blob/aaf5c2092b01219fa0d5c2d323741d409086ca32/slime/backends/megatron_utils/update_weight/update_weight_from_disk.py)、[`slime` 的 SGLang rollout](https://github.com/THUDM/slime/blob/aaf5c2092b01219fa0d5c2d323741d409086ca32/slime/rollout/sglang_rollout.py)、[`verl` 的 GRPO 优势](https://github.com/verl-project/verl/blob/983cb0f24443f87b3d161fad318445130a620b07/verl/trainer/ppo/core_algos.py)。[Megatron Bridge API](https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.models.conversion.auto_bridge.html) 提供 `load_hf_weights` 与 `save_hf_pretrained`；[Qwen 支持列表](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen.html) 包含 Qwen2.5 0.5B。
