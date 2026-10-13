# 两种权重同步方式

当前实现范围：同一 Linux 主机/容器、一张训练 GPU、一张 SGLang GPU、训练和推理 TP=1，
dense Qwen2.5-0.5B-Instruct。实现已接入训练循环，实机验证留待 GPU 环境。

## 直接跨卡传输

```bash
TRAIN_GPU=0 ROLLOUT_GPU=1 bash scripts/run_demo.sh /models/qwen \
  --weight-sync nccl --rollout-dtype bfloat16 --sync-bucket-mb 64 --sync-timeout 120
```

容器入口也接受相同的训练参数。默认仍为 `--weight-sync disk`，两种路径共用 Bridge 转换。

```text
训练 GPU                                           推理 GPU
Megatron BF16/FP32 参数                             SGLang 模型
  │ Bridge.export_hf_weights(cpu=False)                  ▲
  ▼                                                     │
HF 名称与布局 → 转成 rollout dtype → contiguous tensor   │
  │                                                     │
  ├── HTTP: names / dtypes / shapes ─────────────────────┤ 接收准备
  └── NCCL broadcast: tensor 数据 ───────────────────────┘ 写入参数
```

HTTP 负责控制和元数据；真正的 tensor 数据由 NCCL 传递。
训练原有的 WORLD 大小仍是 1，另外建立一个大小为 2 的独立更新通信组：
训练 rank 0，SGLang rank 1。不能用训练 WORLD 上的 `dist.new_group()` 加入推理进程。
这里复用 SGLang 的 `init_custom_process_group`，不复制 PyTorch 私有建组代码。

每次同步顺序：

1. 首次同步时，通过 HTTP 与本地调用并发创建独立 NCCL 通信组。
2. 暂停生成并清空 KV cache。
3. Bridge 流式产出 HF 名称与 tensor，检查名称、shape、有限值及覆盖范围。
4. 按 `sync_bucket_mb` 聚合完整 tensor；先在线程发出 SGLang 接收请求，再广播数据。
5. 等待该批的广播和 HTTP 加载成功，才发送下一批；只有最后一批携带目标版本。
6. 清缓存、查询服务版本，恢复生成后才更新训练侧 `committed_version`。

同步组保持复用，训练结束时两端并发销毁，然后释放 actor 的原有 distributed 状态。
HTTP 和训练侧 NCCL 操作使用 `sync_timeout`；同步失败会退出并保留暂停状态，不自动重试
半次更新，也不在半次更新后切换成磁盘路径。启动器负责最终回收两端进程。

`sync_bucket_mb` 是批次目标大小，不拆开单个 HF tensor。embedding 可能大于该值，
会单独构成一个更大的批次；显存至少要能容纳最大的单个转换结果及传输临时缓冲。
目前每个 tensor 发起一次 broadcast，没有 flatten packing、TP gather 或多节点传输。
初始化使用本机 loopback 的临时端口，两个角色必须处于同一网络命名空间。

## dtype 和版本

actor 默认使用 Megatron 原生 BF16 训练，仍可选择 FP32。启动器将 `--rollout-dtype` 同时用于
SGLang 启动和 NCCL 发送侧转换：默认 `bfloat16`，也可选择 `float32`。
同步发送的是 optimizer 已回写到模型的参数，不是 FP32 optimizer 主参数。
若 actor 为 FP32 而推理为 BF16，传输时发生舍入；相同 dtype 也不保证两端不同内核的
logprob 完全一致，相关数值容差留待实机阶段确认。

当 batch 的优势全为零时不执行 optimizer step、不增加策略版本，也不重复同步。
日志中的 `step` 是采样轮数；`policy_version` 是 actor 已执行更新的次数。
NCCL 路径日志的 `inference_weights` 为 null，因为内存传输不会生成 HF 发布文件。
`training_checkpoint` 指向独立的可恢复训练断点，两种同步路径都会保存。
日志另含 `sync_backend`、`sync_seconds`、`sync_bytes`、`sync_buckets`、`sync_dtype`。
这些文件同步/内存同步入口都不提供训练断点恢复；optimizer/RNG 保存仍属于后续功能。

## 磁盘路径

```bash
bash scripts/run_demo.sh /models/qwen --weight-sync disk
```

先导出完整 HF 文件并检查其结构，再暂停、清缓存、加载目录、核对版本并恢复。
`update_weights_from_disk` 必须明确返回 `success: true`。
版本目录为 `weights/<session_id>/weight_vNNNNNN`，所有版本暂时保留。
每次启动生成独立 session，因此恢复后重新发布同一个策略版本不会覆盖原文件。

实现文件：`nanorl/conversion.py`、`nanorl/sync.py`、`nanorl/sync_nccl.py`。
参考本地 slime 的 `update_weight_from_distributed.py` 与 `hf_weight_iterator_bridge.py`，
以及 [SGLang 接收/加载实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_components/weight_updater.py)。
