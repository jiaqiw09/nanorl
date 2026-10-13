# 采样到 actor 更新的契约

## 样本与分组

`Sample` 包含 prompt/response token、旧 logprob、文本、sample_id、group_id、policy_version
和 finish_reason。每组必须具有相同 prompt、恰好 group_size 个不同样本，整批版本必须与
actor 当前版本一致。优势按 group_id 计算；样本顺序改变不会把不同 prompt 混成一组。

一轮内使用有界并发 HTTP 采样，等待整批完成后才训练/同步。只接受 SGLang 完成状态 `stop` 或 `length`：

- EOS 在 output logprob 中返回时保留在训练 token 内，不对文本重新 tokenize。
- `length` 标记为 truncated，仍按当前生成文本计算规则奖励，并使用已生成 token 训练。
- 空回答、abort、缺失/非有限 logprob、completion_tokens 不匹配、超预算回答、超长 prompt
  都中止当前训练轮，不拼凑不完整分组、不重试后静默替换样本。
- 每次生成前后检查服务版本，若响应还携带 weight_version，也检查它。

版本检查依赖本 demo 独占 SGLang 更新接口且同步执行；不提供异步更新或外部进程改权重时的事务隔离。
HTTP 版本字段证明控制流的版本，不代替权重数值一致性验证。

`--rollout-concurrency` 默认 4，设为 1 可退回串行请求。工作线程数量受限，返回顺序保持原始
任务顺序；分组仍按 group_id。任一请求失败，整批不进入训练，正在执行的请求按 HTTP 超时退出。
reward 在所有回答收集完成后串行执行，自定义 reward 不需要实现线程安全。

## GRPO 和 actor

组内使用样本标准差（分母 G-1）归一化 outcome reward，零方差组得到零优势。
loss 是每条 response 的 token mean，再对所有 response 取均值。
通过可配置 microbatch 累积梯度，每轮一次 optimizer step，没有多 epoch 重复更新。

第一个回答 token 对应最后一个 prompt 位置的 logits，最后一个回答 token 不遗漏。
词表归一化排除 Megatron 为并行对齐添加的 padding 行；prompt 的预测损失不参与更新。
模型 dropout 显式设为零，避免新 logprob 重算引入 dropout 随机性。

更新检查 reward/advantage/logprob/loss 的有限性；在 optimizer.step 前检查 gradient norm。
参数更新后再次检查有限值；失败后 actor 失效，禁止继续更新或导出，需要重新加载。
这是一条失败即退出的路径，不提供 optimizer 回滚。

全零优势 batch 会跳过 optimizer，防止 AdamW 的历史动量在没有本轮信号时继续移动参数。
日志包含 grad_norm、update_applied、policy_version、有效组比例和截断数量。
非零优势并不保证非零梯度或质量提升；不能只用 step/version 递增判断学习效果。

默认已使用 [Megatron DDP + BF16 原生 optimizer](megatron.md)；FP32 + PyTorch AdamW 保留为可选基线。

## Reward 选择

| 参数 | 行为 |
|---|---|
| `--reward exact-integer` | 默认；整个输出必须是正确整数 |
| `--reward last-integer` | 允许说明文字，提取最后一个独立整数比较答案 |
| `--reward my_rewards:score` | 导入可调用函数 `score(example, response_text)` |

例如在 Python import 路径上的 `my_rewards.py`：

```python
def score(example, response_text):
    return float(response_text.strip() == str(example.answer))
```

返回值须能转为有限浮点数；异常或 NaN/Inf 会停止该轮，不默认为零分。
当前数据仍是整数 answer，内置奖励不是通用数学表达式验证器；`last-integer` 也不处理分数、
单位或严谨的推理正确性。添加新的奖励不需要修改训练循环。

## 数据

原来的 `examples/math.jsonl` 保留为极简单例子。另提供难度混合的整数题：

```bash
bash scripts/run_demo.sh /models/qwen --data-path examples/math_mixed.jsonl --group-size 4
```

混合题也不保证 Qwen 每组一定有对有错，日志中的 active_group_fraction 用于观察实际信号。
受控更新入口 `python3 -m nanorl.check_actor ...` 已保留用于后期验证；它使用明确的两个回答
和规则奖励验证更新机制，不把这些受控回答当成真实 rollout 训练效果。当前按功能优先，暂不执行。

## 训练 microbatch

`--train-microbatch-size 2` 表示每次前向/反向处理两个回答，不是两个 prompt 组。
一轮总回答数仍是 `prompts_per_step × group_size`，参数更新只发生一次。
默认 1；如果显存允许可增大，不会自动探测显存或 OOM 后重试。

- 每个 microbatch 右侧补齐至本批最长序列。position_ids 每行从 0 递增，有效 token 的位置不变。
- 目标 Qwen 模型内部使用 causal attention；有效 query 看不到位于右侧的 padding。
  因此 attention_mask 传 None，沿用原生 causal kernel，不引入双向注意力或左侧 padding。
- response_mask 标记用于预测回答 token 的位置：`prompt_length-1` 至 `total_length-2`。
  softmax 前先选出这些位置，排除 prompt、padding 与补齐词表行。
- 每条回答先算 token-mean loss，再将 microbatch 内这些 loss 相加，除以**整轮回答数**后 backward。
  最后一小批不足 microbatch_size 时保持相同分母，不丢弃、不额外放大。
- MCore main_grad 在整轮开始清零，所有 microbatch 后 finalize/clip/step。
  `ratio_mean`、`clip_fraction` 按有效回答 token 统计；clip_fraction 表示 ratio 越出 clip 区间的比例。
  全零优势跳过前向时这两个值为 null，而非伪造测量结果。

数学目标不随 microbatch_size 改变；实际 BF16 累积和不同内核形状仍可能带来浮点误差，尚未做数值比较。

## 可恢复采样器与 seed

`--seed 42` 默认统一 Python、NumPy、torch CPU/CUDA 和 Megatron CUDA RNG tracker。
数据按 epoch shuffle，`--no-shuffle` 可改为原文件顺序。批次可以跨 epoch；数据很小时允许
同一轮再次抽到同一条记录，每次仍分配独立 prompt group。

sampler 保存完整排列、epoch（从 0 开始）、cursor、自己的 Python RNG 状态，以及解析后的
prompt/answer 列表的 SHA256。恢复会检查内容及顺序，数据移动路径不影响恢复。
日志的 dataset_indices 是本轮实际使用的零基索引，epoch/cursor 表示本轮消费完成后的位置。

SGLang 请求 seed 由总 seed 和稳定 sample_id 的 SHA256 派生；线程完成顺序不会改变请求种子。
launcher 同时设置 `--random-seed` 和 `--enable-deterministic-inference`，预检查确认 API 支持。
独立启动服务再运行 `nanorl.train` 时也需开启该开关。该开关可能限制可用内核并影响速度。
此处接法依据 [SGLang SamplingBatchInfo](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/sampling/sampling_batch_info.py)，
仅发送 sampling_seed 而不开启 deterministic inference 不足以启用种子路径。
数据与请求种子恢复不代表跨硬件、依赖版本、并发批形状的 GPU 结果逐位一致。

## Checkpoint 与 resume

默认 `--checkpoint-every 1`：初次同步后保存 step 0；每 N 个完整训练 step 和最后一个 step 保存。
checkpoint 只在采样、reward、更新及需要的权重同步均完成之后提交。全零优势也会推进 sampler 和
训练 step 并按规则存档，但 policy_version 不增加。失败中的半轮状态不存档。

checkpoint 包含：底层 Megatron 模型、完整 optimizer（含 BF16 的 FP32 主参数和 Adam moments）、
completed step、policy_version、全套 RNG、sampler 状态、配置、模型配置及 tokenizer 文件 hash。
使用本地临时文件写完、fsync、原子 rename，最后更新 latest.json；失败不会把半个文件指为 latest。
训练 checkpoint 使用 PyTorch pickle，只加载自己可信的 nanorl 产物。

```bash
# 在 nanorl 目录执行；本例使用默认训练/奖励参数。
RUN_DIR="$PWD/runs/microbatch-demo" bash scripts/run_demo.sh /models/qwen \
  --steps 20 --train-microbatch-size 2 --seed 42 --checkpoint-every 2

# 延长至累计 40 步，可沿用目录；也可指定新的 RUN_DIR。
RUN_DIR="$PWD/runs/microbatch-demo" bash scripts/run_demo.sh /models/qwen \
  --steps 40 --train-microbatch-size 2 --seed 42 --checkpoint-every 2 \
  --resume "$PWD/runs/microbatch-demo/checkpoints/latest.json"
```

`--resume` 接受 `.pt`、`latest.json` 或 checkpoints 目录；`--steps` 是累计目标，不是额外步数。
启动参数仍需给出原训练配置；不一致时列出冲突字段并退出。检查 reward 名称、actor 后端/dtype、
rollout dtype、学习率、clip、组大小、长度、每步 prompt 数、seed 和 shuffle。
模型/tokenizer hash 及数据指纹必须相同；原 HF 权重仍用于构建模型，随后由训练 checkpoint 覆盖。
自定义 reward 函数的源码和内部/外部服务状态不保存，使用者需保持实现及环境一致。

允许修改总 steps、输出目录、checkpoint 间隔、microbatch 大小、rollout 并发和同步方式；
改 microbatch/并发可能改变浮点结果。恢复过程为：

```text
构建 actor → 校验 checkpoint → 恢复 model/optimizer/sampler/version
    → 向 SGLang 发布恢复后的策略 → 恢复训练 RNG → 从 saved_step + 1 采样
```

不复用旧 rollout 或旧 KV cache。遇到更新/同步失败请重启 launcher 并显式 --resume；没有进程内自动回滚。
若 checkpoint 间隔大于 1，崩溃后会重做上次存档以来的轮次。所有 checkpoint 和磁盘发布版本暂时保留，
不自动删除；完整单卡 Adam 断点占用较大，按训练规模选择保存间隔并预留磁盘/CPU 内存。
只支持当前 TP=PP=DP=1、非分片 optimizer，不提供跨并行拓扑或跨 Megatron 版本迁移。

## 持久日志与运行目录

由脚本生成的 RUN_DIR（独立 trainer 中为 --output-dir 的父目录）包含：

```text
RUN_DIR/
├── metrics.jsonl                         追加事件，逐行 flush/fsync
├── sessions/<session_id>.json            每次启动的完整 CLI 配置
├── checkpoints/latest.json              最近完整训练断点指针
├── checkpoints/<session>-step-NNNNNNNN.pt
├── weights/<session>/weight_vNNNNNN/     仅 disk 同步生成的 HF 权重
├── sglang.log                            launcher 追加服务输出
├── environment-{train,rollout}.json       launcher 本次环境报告
└── launch.json                           launcher 本次卡分配/启动配置
```

独立 trainer 默认 --output-dir 为 runs/demo/weights。复用 RUN_DIR 时配置快照及指标保留各 session，
环境报告/launch.json 更新为本次内容；一个目录一次只供一个 trainer 使用。

JSONL 记录 start、resume、initial_sync、checkpoint（初始存档）、step、complete 和 error 事件。
step 含 reward 均值/标准差/范围、有效组比例、loss、grad_norm、ratio、clip_fraction、回答长度和截断数、
采样/reward/更新/同步/存档耗时、版本与数据位置。完整存档路径叫 training_checkpoint；HF 发布目录叫
inference_weights，NCCL 模式后者为空。checkpoint_saved 标记该轮是否实际新存档。
异常记录阶段、最后完整 step、最后可用 checkpoint 和 traceback；SIGKILL/断电无法保证最后一条日志完整。
恢复到较早断点时 step 可能重复，应结合 session_id 解读；可恢复位置以 checkpoint 为准。
文件末尾若因强杀留下残行，下次启动另起一行，读取时跳过损坏行。

不添加独立评估、评估数据集或评估调度。以上为功能实现，按当前要求暂不新增或执行测试，
微批梯度等价性、原生 optimizer 恢复以及两卡中断恢复仍待具备环境后验证。
