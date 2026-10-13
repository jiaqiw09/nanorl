# nanorl 端到端审查与开发清单

## 最新功能进度

按用户最新要求：优先开发功能，测试与 GPU 验证后置；不把未验证的实现写成已验证可运行。

- 03：已接入 sample/group/version/finish_reason 契约，完整组检查、按 ID 计算优势、终止/截断策略、生成前后版本检查。
- 04：已补 causal shift 与词表 padding 处理、dropout=0、有限值和梯度检查、失败 actor 禁止发布、零信号跳过 AdamW。保留受控更新入口供后期使用。
- 06（部分）：已有导出结构检查、显式加载成功确认、提交版本关联样本；Megatron/HF/SGLang 三端 logprob 数值对照仍未开发完整。
- 07：增加按组 ID 的 GRPO、零信号行为和难度混合数据。中断前已写的数值测试保留，后续不继续围绕测试扩展。
- 09：已新增 `sync_nccl.py`，使用 Bridge 流式导出、独立两 rank 通信组、HTTP 元数据与 NCCL 广播、有限 bucket、末批版本提交、通信组清理，CLI `--weight-sync nccl` 已接入主循环。尚未实机执行。

- 10：已新增 `megatron_backend.py`，原生 DDP + FP32 main_grad + Megatron optimizer + BF16 模型/FP32 主参数；每轮显式清零、微步反向、finalize、裁剪/step 与参数回写。训练入口默认此路径，保留 torch/FP32 对照。仍为单 rank，无分片 optimizer、PP schedule 或 LR scheduler。
- 11：已支持 `--rollout-concurrency` 有界并发及 `--train-microbatch-size` 多回答右侧 padding；response mask、position IDs、整轮回答数归一化及短尾批累积已接入。
- 12：已接入 `--reward`，支持 exact-integer、last-integer 与 module:function；奖励异常/非有限值在更新前失败。

- 13：已实现独立训练 checkpoint，保存模型、原生 optimizer/FP32 主参数、RNG、step/version、sampler 及配置；临时文件/原子提交、latest 指针、resume 校验、恢复后初次同步均已接入。按间隔及最终步保存，暂保留全部历史版本。
- 14：已实现配置快照与追加 JSONL，包含 reward/有效组、loss/ratio/clip/grad、长度/截断、阶段耗时、数据位置、策略版本及失败 traceback。**按用户要求移除独立评估范围。**
- 15：已实现 epoch shuffle/顺序采样、seed、完整排列/cursor/RNG 恢复和数据指纹；生成使用固定请求种子，launcher 开启 deterministic inference。

**当前剩余：** 06 三端数值对照和 08 完整两卡验收入口留到实机阶段；本轮四项已开发接入，微批数值、optimizer 状态恢复和中断重启仍未验证。自动版本清理暂不做，历史产物全部保留。TP/DP/PP 扩展不在当前两卡首版范围。本轮按用户要求未新增或运行测试。

## 本轮实施进度（01、02、05）

- 01：已新增 Linux/Python/CUDA/单 rank、目标模型、tokenizer、权重结构和依赖 API 检查；保存实际环境报告；提供仅接受固定镜像 digest 的容器入口。候选 CUDA 镜像仍待实机验证，不能标记为已验证的版本组合。
- 02：已实现 GPU UUID 去重、无效卡与端口检查、服务模型/实例身份校验、独立进程组清理和 actor distributed 清理。补充启动失败、超时、身份错误及实际子 worker 回收测试。两卡完整启动/中断验收待 GPU 主机执行。
- 05：已抽出 Bridge 转换层；新增 checkpoint 完整性检查、独立 Megatron 中间态映射检查和真实 `nanorl.roundtrip` 入口，输出逐 tensor 比较报告与 tokenizer 验证。新增 CPU 数值及损坏检测测试；本机缺少 torch/safetensors，数值测试及真实 GPU 往返未执行。

以下保留原始审查记录，用于对照；其中“现状”指实施前。实现入口和执行方法见 README 与 `docs/`。

实施后本机验证：22 项测试中 14 项通过，8 项因缺少 PyTorch/safetensors 跳过；Python 语法、两个 shell 脚本语法、CLI 帮助检查通过。实际验证了子进程组 SIGTERM 回收、macOS 预检查失败报告、拒绝浮动镜像标签、从项目外调用脚本的路径解析。临时测试依赖安装未获批准，未继续安装。CUDA/Bridge 数值往返、CUDA 内核兼容性及两卡生命周期仍待 GPU 主机验证。

审查范围：现有 8 个 Python 包文件、启动脚本、配置与依赖、样例数据、README 和 2 个测试文件。沿启动、模型初始化、采样、奖励、优势、反向更新、权重同步、退出检查了整个调用链。本轮不修改实现代码。

当前定位：同步、单机两卡、训练 TP=PP=1、Megatron Core 模型 + PyTorch AdamW、HF 磁盘同步的实现原型。不能据此宣称 GPU 端到端可运行。本机缺少 PyTorch/CUDA，下面区分静态发现、开发缺项与尚未执行的 GPU 验收。

已经有的部分：SGLang HTTP 采样、chat template、回答 token 与 rollout logprob、整数奖励、按连续组标准化、token-mean/sequence-mean clipped loss、逐回答梯度累积、Bridge 加载/导出、暂停/清缓存/热加载/版本查询。无 padding 的单样本前向通过切片排除 prompt；当前不需要额外 padding mask。原生 GRPO 不要求 critic；本例省略 reference KL 是明确的算法简化。

上轮已修的两项不重复列为“未实现”：普通 AdamW 路径显式关闭 gradient accumulation fusion；同步失败或版本不符后保持 SGLang 暂停。二者仍需真实运行验证。

## P0：让现有最小闭环可信，优先完成

### 01. 固定运行环境与支持范围

- 现状：`requirements.txt` 只有包名；`config.py` 只检查路径、少量数值和 URL。
- 开发：记录验证过的 Python/CUDA/PyTorch/Megatron Core/Bridge/TE/SGLang 版本；提供可复现的安装或容器说明；启动前检查所需 API、HF 权重与 tokenizer、模型类型和上下文上限。首版明确只支持一种 Qwen2.5 dense 模型与单训练 rank。
- 验收：新环境按文档可初始化两个后端；不支持的模型/并行度在模型加载前报明确错误。没有实机验证前不把随意选定的版本称为已验证版本。

### 02. 启动、非共卡约束与进程生命周期

- 现状：`run_demo.sh` 只比较两个 GPU 字符串；`/health` 成功即可启动训练；退出时只向 SGLang 父 PID 发信号。actor 固定 rank=0/world_size=1，未拒绝外部多 rank 启动，也未显式销毁进程组。
- 开发：校验首版每个角色仅分配一张有效且不同的物理卡；检查端口占用与服务归属、模型身份；避免连到其他已运行的 SGLang；加入可控超时、完整进程回收和 distributed 清理。脚本从自身位置定位项目，消除对当前工作目录的依赖。
- 验收：同卡/无效卡/端口冲突/子进程启动失败均在训练前失败；正常退出、训练异常和中断后不残留本次启动的 GPU worker，也不终止别人的服务。

### 03. 样本数据契约与终止语义

- 现状：`Sample` 只有 prompt_ids、response_ids、old_log_probs、response_text；分组完全依赖列表顺序。忽略 finish_reason；未保留样本 ID、组 ID、权重版本或截断状态。两个长度来自同一 entries 列表，当前长度比较不能独立验证返回 token 是否完整。
- 开发：明确 prompt/group/sample 标识、生成策略版本、结束原因与有效长度；结合服务返回的 completion 信息核对 token/logprob；制定 EOS、长度截断、空回答和超长 prompt 的处理规则；失败不能悄悄造成不完整 GRPO 组。
- 验收：单 token 回答、EOS 终止、长度截断、空输出、超长 prompt 都有测试；组大小与策略版本可验证。当前顺序同步逻辑可以保留；显式标识为后续并发提供依据。

### 04. Megatron actor 更新正确性与数值检查

- 现状：有 backward/clip/step，但没有验证梯度与参数变化；只验证了 rollout 的 logprob 有限。配置实际接受 learning_rate=NaN 和 Inf（本轮已通过直接调用 Config.validate 复现）。
- 开发：验证配置、reward、advantage、loss、梯度的有限性；非有限值时在 optimizer.step 前失败；验证模型 dtype、输出词表范围、causal shift、回答切片和微批梯度累积；记录 grad_norm 和受控测试中的参数变化量。
- 验收：固定的非零优势 batch 产生有限梯度及非零参数变化；首个回答 token 使用最后一个 prompt 位置的 logits，最后一个回答 token 不遗漏；坏数值不会写入新权重。全零优势应明确标记无学习信号，不能仅凭 step 递增声称模型学到了东西。

### 05. 可检查的 HF↔Megatron 转换模块

- 现状：转换封装在 actor 中的两次 Bridge 调用，独立契约与转换测试缺失。Bridge 已实现映射，不能把它说成完全没有转换。
- 开发：抽出针对选定模型的转换适配层，明确 tensor 名称、shape、dtype、QKV/GQA 排列、QKV bias、gate/up 拆分、norm 与共享 embedding 的约定；验证导出配置及权重文件完整性。可以继续复用 Bridge，不要求重复实现其所有映射；学习文档需展开关键变换。
- 验收：真实 HF→Megatron→HF 往返测试检查名称、shape、dtype、数值和 tied-weight 约定；人为破坏关键映射能被测试发现。

### 06. 权重同步成功的判据

- 现状：磁盘导出返回后发送目录路径、加载并核对版本字符串；没有导出内容验证、三端数值比较或样本版本关联。失败后暂停已实现；自动恢复尚无。
- 开发：先验证一个完整的导出版本，再让 SGLang 加载；检查明确的成功响应；把已提交版本关联到后续样本。补固定 token 的 Megatron/导出 HF/SGLang logprob 对照，明确训练与推理 dtype 和容差。首版维持失败即退出并人工恢复即可，不必先做自动回滚。
- 验收：残缺 checkpoint、错误版本或加载失败不能进入下一轮；数值比较通过后才把同步路径判为可用。仅检查 weight_version 不能作为转换或参数一致性的证据。

### 07. 完整的 GRPO 核心测试和有信号样例

- 现状：组内标准化与 clipped surrogate 已有；测试只覆盖一种分组和正优势的一种 clip 情况；PyTorch 测试在本机跳过。样例只有 6 道简单算术题，可能产生全对/全错组。
- 开发：覆盖正/负优势、clip 两侧、组间独立性、零方差组、不同回答长度的 loss 权重与梯度累积；提供受控混合 reward 测试和更合适的小型训练数据；记录有效组比例。样本重排后若要继续分组，需使用 03 的组标识。
- 验收：CPU PyTorch 测试实际执行；受控 batch 必须能产生训练信号。真实采样的 reward 不应伪造，短跑验收不要求准确率必然上升。每轮一次 optimizer step 是合法简化，不强制加入多 epoch。

### 08. 两卡端到端验收脚本

- 现状：测试替换了真实 HTTP 和导出器，模拟导出器仅创建空目录；没有真实模型转换、优化器更新或 SGLang 加载测试。
- 开发：新增固定小模型、固定短输入、有限步数的 GPU smoke 测试，保存环境、配置、逐阶段结果与误差指标。
- 验收：初始化 → 初次同步 → SGLang 采样 → reward/GRPO → actor 更新 → 再次同步 → 下一轮采样至少完整执行一次；同时用受控更新确认参数变化，并让 SGLang 固定 continuation 的 logprob 与更新后策略一致。追加一次加载失败用例验证停止行为。

## P1：补齐用户关心的机制和基本可用性

### 09. NCCL 内存权重传输

- 现状：只有共享磁盘方案；HTTP 携带路径，不携带 tensor。
- 开发：在复用 05 的转换结果后，建立训练进程到 SGLang 的独立更新通信组；发送 names/dtypes/shapes，按有限 bucket 广播 tensor；接收请求与广播正确并发启动，避免双方等待；完成同步后清缓存、确认版本并恢复生成。首版只做一张训练卡到一张推理卡。
- 验收：无需每步落完整 HF 文件也能同步；数值测试与磁盘路径使用同一组固定 token。测量字节数和耗时。NCCL 是为展示直接跨卡传输而补的功能，并非现有磁盘方案正确性的前提。

### 10. Megatron 原生训练组件与 BF16 路径

- 现状：模型是 Megatron Core，优化器是原生 PyTorch AdamW，参数 FP32，没有 MCore DDP/optimizer/schedule 集成。
- 开发：若目标包含学习 Megatron 训练后端，加入一条明确的 MCore DDP + Megatron optimizer 路径，处理 BF16 模型参数、FP32 optimizer 状态、梯度缓冲和微批调度，先在单 rank 验证。不要只替换 optimizer 类名。
- 验收：单步更新、梯度缩放与权重导出正确；记录显存和数值误差。现有单 rank AdamW 可以作为最小基线，缺少这些组件本身不意味着它必然错误。

### 11. Rollout 并发与训练 microbatch

- 现状：`train.py` 对每个回答发一次串行 HTTP 请求；actor 每次只前向一个回答。
- 开发：先增加有界 SGLang 并发并维持完整分组，再增加可配置 microbatch；引入 padding 后明确 attention mask、response mask、position_ids 与 loss 分母，保证和单样本基线等价。
- 验收：并发完成顺序不改变优势分组；microbatch 大小变化后梯度在容差内一致；padding 与 prompt 不参与策略 loss。

### 12. 可选择的 reward 实现

- 现状：有 Reward Protocol，但训练入口硬编码 ExactIntegerReward，数据 answer 必须是整数。
- 开发：提供一种简单的配置选择/函数加载方式；明确 reward 输入输出、有限值与异常规则，保留可解释的得分信息。首版仍只需规则奖励，不需远程奖励服务或神经 reward model。
- 验收：增加第二个规则奖励时不修改训练循环；解析失败、格式失败和答案错误可辨别，不混为运行成功。

### 13. 训练 checkpoint 与 resume

- 现状：导出的 HF 文件用于推理；没有 optimizer 状态、训练步、RNG、数据位置及恢复入口。输出目录同版本碰撞会失败，所有权重版本一直保留。
- 开发：区分推理发布文件和可恢复的训练 checkpoint；保存 actor、optimizer、step、RNG、数据采样状态和配置；恢复后先把正确版本同步给 SGLang；设定简单的版本保留策略。
- 验收：中断恢复与连续运行的下一批数据和一次受控更新可比较；恢复不能把旧 rollout 喂给新版本 actor；清理不删除仍在使用的 checkpoint。

### 14. 日志与诊断（独立评估已移出范围）

- 原始现状：控制台仅打印 mean_reward、loss、nonzero_advantages、checkpoint；没有持久训练指标和阶段耗时。
- 开发：本地 JSONL 保存配置/环境与指标；包括 reward 分布、有效组比例、回答长度/截断率、ratio/clip_fraction、grad_norm、采样/更新/同步耗时及版本。不添加评估数据和评估调度。
- 验收：能区分没有训练信号、更新失败、同步耗时和推理数值偏差。无需先接入外部实验平台。

### 15. 数据采样与随机性管理

- 现状：读取整个 JSONL，按原顺序无限 cycle；没有显式 seed、shuffle、epoch 或 sampler 状态。
- 开发：加入简单且可复现的 sampler，统一 Python/torch/生成侧可用的随机种子，记录数据标识；与 13 的恢复状态衔接；明确超长输入的拒绝或截断策略。
- 验收：相同设置可重放样本顺序；断点恢复不重置数据位置。GPU 内核的数值非确定性与数据顺序可复现需要分别说明。

## 后续可选，不计入首版完成标准

- 训练 TP/DP，多卡梯度同步、词表并行 logprob 与转换时的 shard 聚合；PP/多节点放在更后面。
- rollout/training 异步重叠、旧策略样本管理和版本落后限制。
- reference KL、神经 reward model、多轮 agent rollout、MoE、量化等；按明确需求增加。
- Ray、通用插件注册系统或复杂调度器不作为这个两卡入门 demo 的前置依赖。

## 建议开发顺序

01 环境 → 02 启动 → 05 转换往返 → 04 受控 actor 更新 → 03 样本契约与 07 GRPO 测试 → 06 同步数值检查 → 08 两卡验收。

现有磁盘闭环通过后，优先实现 09，展示用户关心的跨卡推权重；10 按 Megatron 后端学习深度推进。11–15 逐项补齐性能、可恢复性和可观测性。

本轮验证：重新通读所有现有文件；直接调用配置校验确认 NaN/Inf 学习率被接受。沿用上轮未变更代码的测试记录（6 项通过，1 项因缺少 PyTorch 跳过），不把模拟测试或静态审查算作真实 GPU 验收。
