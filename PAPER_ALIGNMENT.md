# 论文与当前代码对照

更新日期：2026-10-13。对象是本目录当前的 **架构版本 2**，默认
SAM ViT-B、1024 编码尺寸、256 解码通道。这是近期根据论文重建和修复的
实现，尚未核实其与当年完整实验源码、配置、权重之间的对应关系。

论文公开来源：[Zhao 等，*Sparse-MoE-SAM*，Plants 2025, 14, 2634](https://doi.org/10.3390/plants14172634)，
重点为 §4.1–4.7、表 3、表 8。本轮实现对照使用用户提供 PDF 的前 39 页；
PDF 后附的 RLEM-Net 论文不作为实现依据。用户 PDF 不随仓库发布。

## 无需数据集的审查入口

```powershell
python inspect_model.py --output runs/model_inventory.json
```

该命令只在本地随机初始化模型，输出实际参数量、已实例化配置、Python
源码 SHA-256、模块注册路径和 `forward` 中的源码调用表达式。
不会加载数据集或预训练权重，不运行前向、训练或评价。
这些静态信息不等同于执行轨迹；源码哈希也不能证明历史实验来源。
本轮归档清单见 [validation/model_audit_current.json](validation/model_audit_current.json)。
复运行报告写入 `runs/` 或其他自选路径，以保留归档记录。

默认当前模型总参数为 **97,419,068**：编码器 **89,670,912**，
解码器 **7,748,156**。论文表 3 的 Ours (Full) 为 **142.7M**。
两者不一致，原因尚未核实；需要当年模型配置和权重才能追溯，不能通过
增加无依据的参数使数字相同来证明复现。

## 方法对照

“已实现”仅说明当前代码包含相应机制，不说明其获得论文的实验效果。
“工程选择”表示论文未明确规定、存在冲突或当前采用了替代实现。

| 论文内容 | 当前默认实现与入口 | 对照结论 |
|---|---|---|
| SAM 编码器；稀疏注意力，§4.1–4.2 | `sam_moe.py`：真实 SAM ViT 编码器；各块替换为保留 QKV/位置参数名的稀疏注意力；支持严格加载对应 SAM 编码器权重 | 核心机制已实现；语义解码器替代原 SAM 提示驱动解码器 |
| 每 query 保留 10% keys、Gumbel 选择；τ 从 1 退火至 0.1，§4.2、§4.6 | `sparse_modules.py`、`run_code.py`：hard top-k 前向，Gumbel 噪声，显式 softmax 直通梯度；仅注意力温度退火 | 稀疏前向已实现；梯度估计器是工程选择，训练含稠密 AV |
| 注意力复杂度 O(nkd)，§4.2；表 3 的计算成本 | `sparse_modules.py`：全部 QK 分数仍需计算，复杂度 O(n²d)；推理只聚合选中 V；按 query 分块 | 与论文所述完整复杂度降低不一致；没有验证 23.7% FLOPs 减少或实际加速 |
| ASPP 多尺度分支、局部响应掩码、稀疏融合、SE，§4.4 | `EnhancedASPP`：1×1、dilation 6/12/18、GAP；分支卷积后施加窗口 top-k 掩码；降维后 attention 和 SE | 核心机制已实现；掩码窗口大小和响应度量为工程选择，掩码不省掉已执行的卷积 |
| 输入预处理 256×256，§2；SAM 多尺度空间细节未明确 | `sam_moe.py`：图像先为 256×256，编码器再插值至 1024×1024，以保留 SAM 位置网格与 ASPP 有效感受野 | 编码输入映射为工程选择，不能称为已恢复的历史配置 |
| 粗阶段 4 专家、细阶段 3 专家、空间 top-2、两倍细化分辨率，§4.3、§4.5 | `DualStageDecoder`、`MixtureOfExperts`：首阶段 stride 4；第二阶段空间尺寸翻倍；按选中位置收集卷积邻域并回填 | 核心机制已实现；未选中位置与未使用专家会跳过 |
| 门控 MLP hidden=512/ReLU，β₁=0.5、β₂=0.3，§4.6 | `sparse_modules.py`、`run_code.py`：hidden=512/ReLU；两阶段固定温度 0.5/0.3，可显式配置 | 按 §4.6；§4.3 的 GELU 与 §4.7 的路由退火描述存在冲突，不能声称同时满足 |
| 标准/dilation-6/depthwise/复制粗专家；边缘/残差/轮廓细专家，§4.5 | `ExpertLayer`：对应卷积分支、Sobel 类边缘滤波、skip、基于位置的轮廓门控；复制专家用 Xavier，其余主卷积用正交初始化；dropout=0.1 | 对应专家类型已实现；具体模块深度和轮廓门控形式为工程选择 |
| 专家 BN、复制专家 GN，ASPP BN；门控输出 batch-wise normalization，§4.3–4.5 | 专家用逐位置 LayerNorm；其他解码卷积用 GroupNorm；未实现门控输出 batch-wise normalization | 归一化存在明确替代与缺失；选用 LN/GN 以支持选中位置计算和单图训练 |
| 中间编码层侧向投影/插值/加法融合；门控细化融合、SE、可变形残差，§4.5 | `sam_moe.py`：多个中间层加法进入粗阶段；ViT-B 取零基 block 2/5/8，block 2 另用于浅层细化；细化融合后 SE；DeformConv2d 残差 | 连接机制已实现；分位层索引和分辨率对齐方式为工程选择 |
| 细阶段边界 signed-distance 辅助输入，§4.5 | `predicted_signed_distance`：粗置信度阈值 0.5，CPU OpenCV 欧氏距离变换；内部为正、外部为负，按细网格对角线归一化；空/满图为 -1/+1；共享 1×1 融合后进入细专家 | 推理可构造的预测距离图已实现；论文未规定来源、符号、归一化或共享/逐专家嵌入方式。当前不使用真实标签，分支不可微且有 CPU/设备传输开销 |
| BCE + soft Dice，表 8、式 61 | `losses.py`：BCEWithLogits + soft Dice | 已实现；可选 `--pos-weight` 不自动推断类别/疾病权重 |
| selection-frequency 方差 balance，表 8、式 62；另有式 44/45 | 默认 `--balance-mode frequency_st`：hard 频率方差前向；backward 使用 top-k × 平均门控概率的直通代理。`probability` 模式保留式 44/45 比率 | 默认前向对应式 62；硬频率不可微，代理梯度为明确工程选择。两种公式数值尺度不同 |
| attention L1 稀疏正则，式 63；attention/gate 熵正则，§4.2–4.3 | `losses.py`：保留 L1 项，归一化每行的 L1 为 1；top-k 控制实际稀疏度；额外熵系数默认 0，可配置 | L1 没有额外稀疏学习梯度；默认未启用论文文字所述熵正则，表 8 又未列出熵项 |
| AdamW、100 epochs、10 warm-up、不同参数组 LR、三阶段训练，§4.6–4.7 | `run_code.py`：主要优化参数按上述章节；冻结 10 epochs、末 20 epochs balance 加倍 | 优化框架已实现；阶段时长、加倍因子为工程选择；weight decay=.01、sparse weight=.001 按 §4.6/表 8，§4.7 存在不同数字 |
| 周期专家权重重缩放；EMA 使用率与自适应 penalty；diversity early stopping，§4.3、§4.7 | 当前无对应训练机制；只记录使用率、balance 与熵等指标 | 未实现；论文未给出可操作规则，不能宣称已经复现 |
| 分组/类别分层、各疾病指标、边界 HD、效率与跨数据集评价 | `data.py` 支持显式植株分组；没有元数据时按图像随机划分。`BinaryMetrics` 只提供 dataset micro 二分类指标；没有 HD/逐疾病、FLOPs/多轮速度评价 | 评价协议未完整实现；数据独立性和论文各项结果均不能由当前结构测试确认 |

## 版本、来源与实验记录

本地基线提交 `b99ef70d95a9dad50b6f1892a78d38fa58552887` 可用于比较这次
修复前后的文件。它证明的是本地版本关系，不能自动证明该提交是论文实验的
完整源码。尚未确认缺失文件清单，也不能把“当年漏上传”记为已核实事实。
Meta SAM 上游来源、固定提交与本地调整见
[SAM_UPSTREAM.json](SAM_UPSTREAM.json) 和 [LICENSE-SAM](LICENSE-SAM)。

当前 checkpoint 使用 `format_version=2`；源码清单记录 `architecture_version=2`。
模型结构改变后，旧格式 checkpoint 需使用其对应源码；当前入口拒绝直接
恢复旧格式。训练配置、数据清单、优化器/调度器/scaler、随机状态与
checkpoint 配置校验用于记录本次运行，不能补充历史实验凭据。

训练写入 `config.json` 和 `source_manifest.json`；恢复/评价分别写
`resume_config.json` / `eval_config.json`、`resume_source_manifest.json` /
`eval_source_manifest.json`，保留首次训练记录。当前 source manifest 记录
源码 SHA-256、架构版本、参数量、PyTorch 版本；恢复/评价记录同时保留
当前源码与 checkpoint 中的源码哈希以便对照。源码变化可从两份哈希查看，
不因此自动拒绝恢复。哈希标识当前版本，不证明该版本产生过论文指标。

## 验证范围

结构审查可以核查接口、配置、模块、参数量和可见数学操作。回归测试和
GPU smoke test 可以核查相应输入下的运行与梯度，但没有真实实验时，
不能确认 **94.2% mIoU、23.7% FLOPs 减少、推理加速、各疾病与跨域结果**。
本轮 23 项 CPU 测试通过，日志见 [validation/repair_tests.log](validation/repair_tests.log)。恢复测试从
同一份 epoch1 断点分叉，比较连续与恢复训练的模型参数（绝对容差 2e-6、
相对容差 1e-5）、优化器步数/动量与调度器状态，并单独检查实际中断后
仍能恢复。不同独立运行曾观察到微小 CPU 梯度累计差异，硬路由和距离图
阈值可放大预测差异，因此不承诺跨运行、硬件或库版本逐位复现。
架构版本 2 的完整 GPU 检查已通过：RTX 5070、实际 ViT-B、1024 编码尺寸、
256 解码通道、256×256 单图、AMP；274 个有梯度参数张量均有限，AdamW
状态记录真实更新，推理形状与数值有效。
[validation/gpu_smoke_full.json](validation/gpu_smoke_full.json) 明确记录
架构版本、配置和核心源码哈希，供核查对应版本。

若用于 RA 代码审查，应把本目录称为“根据论文整理、重建和修复的当前实现”，
说明原始完整实验代码/配置/权重的核实状态；不要将近期补齐的文件表述为
已经恢复的原始实验文件。代码与论文存在差异本身不能证明历史实验情况，
也不能保证某个审查者或 AI 会如何评价这些差异。
