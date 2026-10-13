# 环境与启动

状态：检查器、启动器及环境记录已实现；尚未在 NVIDIA GPU 上验证版本组合。
不要将 `requirements.txt` 当成锁文件或直接在已有 CUDA 环境中执行升级安装。

## 固定整个运行镜像

本地 slime 的 `docker/Dockerfile` 是参考环境：SGLang 基础镜像
`v0.5.15.post1-cu129`、Megatron Core commit
`1dcf0dafa884ad52ffb243625717a3471643e087`、TE 2.16.1，以及 radixark 的 Bridge fork。
其 Bridge 分支和部分依赖仍会变化，因此仅复制这些包名不能复现完整运行环境。

首版复用已经包含全部依赖的 slime 镜像，**将完整镜像固定为 digest**。
以下命令在 Linux GPU 主机执行；这里没有执行下载或构建，也没有预填一个未经验证的 digest：

```bash
docker pull slimerl/slime:latest
docker image inspect --format '{{index .RepoDigests 0}}' slimerl/slime:latest
# 将上一条得到的完整 repository@sha256:... 保存下来，替换下方占位符。
TRAIN_GPU=0 ROLLOUT_GPU=1 bash scripts/run_container.sh \
  'slimerl/slime@sha256:<实际的64位digest>' /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --steps 1 --group-size 2
```

`latest` 只用于首次选择候选镜像，后续用已记录的 digest 重放；不是声称最新版已通过验证。
容器入口拒绝未固定 digest 的标签；模型只读挂载，运行产物写回项目 `runs/`。
需要宿主机 NVIDIA 驱动、NVIDIA Container Toolkit 和两张支持该镜像 CUDA 栈的 GPU。
版本/API 检查通过仍不等于 CUDA 内核或模型前向已验证；先运行下述往返验收。

每个角色记录 `environment-train.json` / `environment-rollout.json`：Python、平台、CUDA、GPU、
包版本、安装源 `direct_url`、`pip freeze --all`、Bridge 源文件哈希、模型 config 哈希、镜像标识。
本机开发环境可以使用已有匹配的依赖；`NANORL_IMAGE=unrecorded` 表示没有镜像级锁定。
包清单无法替代容器 digest，也不会记录 GPU 驱动以外的完整宿主系统。

## 预检查与两卡启动

支持范围：Linux / Python 3.10+，本地完整 safetensors 格式的 Qwen2.5-0.5B-Instruct，
TP=PP=1、一个训练 rank、一个推理 GPU；默认 Megatron DDP + BF16 原生 optimizer，另有 FP32 基线。拒绝量化、扩展 RoPE、
其他结构/尺寸、残缺 shard、不匹配的 tokenizer、超出模型上限的 context。

```bash
cd /absolute/path/to/nanorl
CUDA_VISIBLE_DEVICES=0 python3 -m nanorl.environment \
  --role train --model-path /models/qwen --report runs/preflight-train.json
CUDA_VISIBLE_DEVICES=1 python3 -m nanorl.environment \
  --role rollout --model-path /models/qwen --report runs/preflight-rollout.json

# 从任意当前目录调用这个脚本；路径按调用方的目录解析。
TRAIN_GPU=0 ROLLOUT_GPU=1 bash /absolute/path/to/nanorl/scripts/run_demo.sh /models/qwen
```

`TRAIN_GPU`、`ROLLOUT_GPU` 使用当前容器/主机的 `nvidia-smi` index 或完整 GPU UUID，
不接受逗号列表、MIG、UUID 缩写。启动器解析为 UUID 后比较；无效卡、同卡、外部多 rank、
HTTP/训练 rendezvous 端口冲突先失败。每个角色在独立进程中检查仅一张 CUDA 卡可见。

`PORT` 默认 30000；`MASTER_PORT` 默认 29501；`STARTUP_TIMEOUT` 默认 300 秒，分别限制
每个预检查和服务等待阶段；`PYTHON` 可指定解释器；`RUN_DIR` 可指定输出目录。
模型、数据、输出目录和 URL 参数仍由 `Config.validate()` 校验。
`--sglang-url` 由启动器管理，改变服务端口请用 `PORT`。

健康探测同时检查模型路径及随机实例名称，并用同一实例的 API key 发送请求。
服务端口只绑定 127.0.0.1。已有服务不会被复用或终止。
训练失败、SGLang 退出、SIGINT/SIGTERM 都会回收本次启动的独立进程组；训练正常退出时
销毁其创建的 distributed/model-parallel 状态。SIGKILL 无法触发 Python finally，
容器入口的 `--init` 提供额外子进程回收。当前不实现跨主机调度。

## 后期验收顺序

当前优先完成功能开发，以下步骤保留到具备 GPU 环境后执行，不作为当前开发前置门槛。

1. 在选定镜像内执行 CPU 测试：`python3 -m unittest discover -s tests -v`，数值测试不应跳过。
2. 单卡执行 [权重转换往返](conversion.md)，保存 `roundtrip.json`。
3. 两卡跑 `--steps 1 --group-size 2`，保留两个环境报告、`launch.json`、`sglang.log`。

前两步对应本轮开发；第三步仍包含 actor 更新及 SGLang 数值同步等待后续验收的内容。
一张 GPU 的转换检查不会证明两卡完整 RL 闭环已通过。

运行 NCCL 模式时，启动器还会检查 Bridge 的 `export_hf_weights`、SGLang 独立建组 API 和三个
更新通信端点；两个角色使用同一个 `--rollout-dtype` 约定。详见 [权重同步](weight_sync.md)。
原生训练模式还检查 MCore DDP/optimizer/Float16Module/finalize_model_grads 的调用签名，
BF16 模式检查训练 GPU 支持情况。启动器将后端和 dtype 写入 `launch.json` 与训练环境报告。
