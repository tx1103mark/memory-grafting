# Qwen3 Base Memory Grafting 实验总结

更新时间：2026-09-15。本文汇总仓库内已完成的训练、消融和评测。所有实验以 Qwen3-0.6B-Base 为学生、Qwen3-8B-Base 为教师，使用无标签中文文本进行 causal language-model continued pretraining，最终通过 `lm-evaluation-harness==0.4.13` 零样本评测 C-Eval validation 和 CMMLU。

## 1. 研究问题

实验检验：将教师模型对高频 2/3/4-gram 的离线 hidden state 存为确定性查找表，并注入学生某一 Transformer block 后，能否提高学生的中文知识能力。

初版注入为：

\[
h'_l=h_l+\alpha\,I_{hit}\,\sigma(\langle RMS(h_l),RMS(W_km)\rangle/\sqrt d+b)W_vm
\]

其中教师表冻结，只训练 K/V 投影、gate、残差系数和学生 LoRA。G 是正确教师表，S 是打乱 key-row 对应的教师表，R 是匹配统计量的随机表，L 是仅 LoRA，B0 是未训练 Base。

机制修复版把表扩至 500k entries；未命中位置由两个 65,536 bucket、128 维的可训练 2/3-gram hash embedding 提供 fallback；注入后增加 kernel=4 的零初始化因果 depthwise ShortConv。显式 adapter 开关用于 memory on/off 评测，避免“清零 memory id”错误地落入 fallback。

## 2. 数据与评测协议

- 原料：Ultra-FineWeb 中文分片，无人工标签；文本经 Qwen3 tokenizer 转成 token IDs，标签为右移一位的下一个 token。
- 清洗：中文比例和长度筛选、规范化精确去重、MinHash 近似去重，以及针对评测题干和选项的字符串重叠排除。
- 初版表：每阶 20k，共 60k 条 2/3/4-gram；按训练文档频次选择。
- 机制修复表：共 500k 条，2/3/4-gram 分别为 166,667/166,667/166,666 条；由 49,757 篇文档生成 75,199 个长度不超过 1,024 的训练 chunk。
- 评测：Base 模型、0-shot、无 chat template；C-Eval validation 用于开发和配置选择，CMMLU作为较独立的确认集。
- 汇总分数默认采用学科 macro accuracy；逐题加权 accuracy 保留在原始 JSON 中。

## 3. 已完成的消融

| 阶段 | 变量 | 训练规模 | 主要目的 |
|---|---|---:|---|
| Pilot | B0/L/R/G、memory on/off | 2M | 验证端到端代码和支路是否参与计算 |
| Gate | 初始 alpha/gate 强度 | 2M | 排除零初始化或小 gate 导致支路学不到 |
| Layer | 教师 T4/T6/T8/T12 × 学生 S1/S3/S6/S12 | 2M | 搜索教师来源层与学生插入层 |
| Long | 7 个层位组合，G/R/L | 10M，含 2M/5M 快照 | 判断更长预算能否形成教师优势 |
| Semantic | G/S/R/L，T12→S1 与 T8→S12，seeds 42/43/44 | 10M | 用 shuffled teacher 分离教师语义与结构效应 |
| Alignment probe | 正确/打乱教师行到学生 hidden 的线性映射 | 20k keys | 训练前判断教师 latent 是否可对齐 |
| Mechanism repair | 500k 表、fallback、ShortConv，G/S/L | 5M，含 0.5M/1M/2M 快照 | 检验更接近完整机制的实现 |

## 4. 关键结果

### 4.1 10M 层位实验

初版 60k 表的七个配置中，10M 时只有 T12→S1 的 G−R 为正（+0.196 个百分点），且置信区间跨零。其余六个教师表不优于随机表。门控没有关闭：实际注入约为 hidden norm 的 0.59%–1.70%。这说明模型使用了 memory 支路，但没有证明教师语义迁移。

### 4.2 三种子 shuffled-teacher 确认

| 配置 / 任务 | G | S | R | L | G−S |
|---|---:|---:|---:|---:|---:|
| T12→S1 C-Eval | 50.552±0.357 | 50.538±0.541 | 50.304±0.512 | 50.484±0.355 | +0.014 |
| T12→S1 CMMLU | 49.781±0.155 | 50.452±0.196 | 50.296±0.146 | 50.603±0.090 | **−0.671** |
| T8→S12 C-Eval | 50.503±0.403 | 50.163±0.575 | 50.630±0.546 | 50.484±0.355 | +0.340 |
| T8→S12 CMMLU | 50.580±0.216 | 50.474±0.010 | 50.610±0.117 | 50.603±0.090 | +0.106 |

正确表没有稳定优于打乱表。T12→S1 在 CMMLU 上三个 seed 均更差，因此初版结构的收益不能归因于教师知识。

### 4.3 Alignment probe

| 配置 | 正确表 cosine | 打乱表 cosine | 正确表 CKA | 打乱表 CKA |
|---|---:|---:|---:|---:|
| T12→S1 | 0.8132 | 0.4932 | 0.00665 | 0.00212 |
| T8→S12 | 0.9088 | 0.6637 | 0.11141 | 0.00002 |

独立的相同配方线性映射能够从正确教师行更好地预测学生 hidden，尤其是 T8→S12。这表明教师 latent 并非不可对齐，问题更可能出在注入接口、覆盖率和训练目标。

### 4.4 500k + fallback + ShortConv 机制修复

单 seed（42）结果如下，分数为百分比：

| 配置 | C-Eval 0.5M | 1M | 2M | 5M | CMMLU 5M |
|---|---:|---:|---:|---:|---:|
| L | 49.50 | 49.39 | 49.46 | 50.80 | 50.82 |
| G T8→S12 | 49.47 | 49.20 | 49.99 | 50.83 | 50.74 |
| S T8→S12 | 49.85 | 49.52 | 50.64 | 50.06 | 50.88 |
| G T12→S1 | 49.00 | 49.66 | **51.87** | **52.16** | 50.17 |
| S T12→S1 | 49.33 | 49.15 | 51.20 | 51.13 | 50.85 |

T12→S1 在 C-Eval 5M 上比 L 高 1.36 点、比 S 高 1.03 点，并从 2M 起保持优势。诊断中，正确表关闭 memory 后 hit-token loss 上升约 0.000589；打乱表关闭后 hit-token loss 略有改善。这是目前最清晰的教师对应关系信号。

CMMLU 没有复现：T12→S1 G 比 L 低 0.65 点，T8→S12 接近持平。因此这仍是开发集上的单 seed 积极信号，不能作为稳定中文知识提升的结论。

## 5. 当前结论

1. 离线 frozen teacher memory 可以低成本接入现有 Qwen3-0.6B-Base，训练和 `lm-evaluation-harness` 评测链路可运行。
2. 简化的 60k 单层接口会使用 memory，但 G/S/R 无稳定差异，无法证明教师语义迁移。
3. 教师向量和学生表征在线性 probe 中存在可利用对应关系；T8→S12 的表征对齐最明显。
4. 扩表、fallback 和 ShortConv 后，T12→S1 在 C-Eval 出现随训练预算增长的 G−S 优势，说明机制修复方向值得继续。
5. 该优势未跨到 CMMLU，且机制修复版只有一个 seed。当前证据支持“存在任务相关信号”，尚不支持“稳定提升中文能力”。

### 4.5 三种子确认与组件归因

预注册确认实验已经完成，详细结果见 [CONFIRMATION_RESULTS.md](CONFIRMATION_RESULTS.md)。CMMLU 主指标上，full G/S/L 分别为 `50.122±0.682 / 50.829±0.020 / 50.902±0.101`，G−S 三个 seed 全为负，均值 −0.707 点。C-Eval 上 G−S 均值仅 +0.238 点，并在 seed 44 反转为 −1.176 点，上一轮单 seed 信号未稳定复现。

组件消融显示，CMMLU 上完整机制比关闭 fallback 低 0.138 点，比关闭 ShortConv 低 0.177 点，三个 seed 方向一致；关闭教师表、仅保留 fallback 得到 50.866，接近 S/L 且方差更低。因此当前教师表不仅没有提供稳定知识收益，还引入了明显的训练方差。

按预注册决策，本项目不再为相同 causal continued-pretraining 目标追加 token 或扫描更多层位。下一阶段转向显式 alignment pretraining，以及 teacher table 的 whitening/contrastive projection；只有离线 held-out key 上正确表稳定优于 shuffled，才重新进入下游训练。

### 4.6 显式 Alignment 改变 CMMLU 结果

显式 alignment 已完成，详见 [ALIGNMENT_STUDY.md](ALIGNMENT_STUDY.md)。离线 held-out keys 上，mean-centered teacher table 经 cosine+InfoNCE 投影后，正确映射 Recall@10 达 99.56%–99.61%，shuffled 仅 0.49%–0.59%。Whitening/PCA 没有优于简单 mean-centering。

将 1024 维对齐表接回学生并关闭 fallback/ShortConv 后，2M-token 的 low-LR G 在 CMMLU 上达到 `51.100±0.029`，相比 low-LR S 高 `+0.238`、相比 L 高 `+0.229`；两个差值在三个 seed 上均为正。Frozen G 相比 L 也稳定高 `+0.179`。使用同一对齐表但随机初始化投影的 G 为 50.951，低于 low-LR aligned G。

因此此前失败的主要原因更可能是教师 latent 与学生 residual space 缺少稳定映射，而非教师表本身没有信息。显式 alignment 将 CMMLU 的 G−S 从未对齐版本的负值转为小但跨 seed 一致的正值。C-Eval 没有同步提升，当前结论限定为约 +0.23 个百分点的 CMMLU 信号，仍需 5M-token 确认。

## 6. 代码与结果索引

- 核心模型：[graft/model.py](graft/model.py)
- n-gram 查表与数据拼接：[graft/data.py](graft/data.py)
- Harness 适配：[graft/harness.py](graft/harness.py)
- 数据处理：[scripts/prepare_data.py](scripts/prepare_data.py)
- 500k 扩表：[scripts/expand_keys.py](scripts/expand_keys.py)
- 教师表构建：[scripts/build_memory.py](scripts/build_memory.py)
- 训练与评测：[scripts/train.py](scripts/train.py)、[scripts/evaluate.py](scripts/evaluate.py)
- 对齐探针：[scripts/probe_alignment.py](scripts/probe_alignment.py)
- 机制实验调度：[scripts/run_mechanism_study.py](scripts/run_mechanism_study.py)
- 原始汇总：[remote_results](remote_results)
- 分阶段报告：[PILOT_RESULTS.md](PILOT_RESULTS.md)、[LAYER_RESULTS.md](LAYER_RESULTS.md)、[LONG_RESULTS.md](LONG_RESULTS.md)、[SEMANTIC_RESULTS.md](SEMANTIC_RESULTS.md)
- 三种子机制确认：[CONFIRMATION_RESULTS.md](CONFIRMATION_RESULTS.md)

模型权重、原始数据、冻结 memory table、checkpoint 和逐题 sample 文件因体积或数据授权原因不进入 Git；仓库保留生成脚本、manifest、汇总 JSON 和结论文档。
