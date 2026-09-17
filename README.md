# Memory Grafting for Qwen3

**Open research on transferring offline n-gram representations from Qwen3-8B-Base into Qwen3-0.6B-Base.**

本项目研究一种轻量的外部记忆注入方法：预先抽取大模型对高频短语的 hidden state，保存为冻结查找表，再通过可训练投影和 gate 注入小模型。仓库公开训练代码、消融设计、评测结果和失败结论，重点回答教师表示是否真的带来超出 LoRA 与额外参数的收益。

> [!NOTE]
> **TL;DR：** 学生模型能够识别并使用正确的教师 memory，显式 alignment 也能把教师表示可靠映射到学生空间；但 2M tokens 时约 `+0.23 pp` 的 CMMLU 信号没有在预注册 5M 实验中复现。当前证据支持“局部语言建模有效”，尚不支持“稳定提升中文知识能力”。

如果这个项目对你的研究有帮助，欢迎通过 Issue 讨论新的对照实验或复现结果。

## 📢 Latest Updates

- **2026-09-17** — 重构项目文档，汇总完整实验路径、术语解释和最终证据边界。
- **2026-09-16** — 完成 aligned memory 的 5M-token 预注册确认：G−S `−0.075 pp`，未复制 2M 正向信号。
- **2026-09-16** — 完成显式 alignment 与 2M 下游实验：low-LR G−S `+0.238 pp`，三个 seed 均为正。
- **2026-09-15** — 完成三 seed semantic control、机制修复与组件消融。
- **2026-09-14** — 完成端到端 pipeline、gate、层位和 10M-token 长预算实验。

## 🔍 Quick Navigation

- [Key Finding 1：教师与学生表示可以对齐](#key-finding-1)
- [Key Finding 2：2M 信号没有通过 5M 确认](#key-finding-2)
- [Key Finding 3：模型使用了 memory，但收益停留在局部 loss](#key-finding-3)
- [实验路线](#experiment-roadmap)
- [快速开始](#quick-start)
- [复现实验](#reproduce)
- [术语说明](#terminology)
- [文档与结果](#documentation)

## 📖 Introduction

Memory Grafting 将预训练模型视为离线 memory constructor。对训练语料中的高频 2/3/4-gram，我们用 Qwen3-8B-Base 编码短语，并保存最后一个 token 的中间层表示。Qwen3-0.6B-Base 在指定 block 后执行最长后缀精确匹配，再将取回的 memory 投影、门控并加入 residual stream。

```text
tokens ── exact longest-match lookup ── frozen teacher table
   │                                           │
   └── Qwen3-0.6B block l ── query-key gate ──┤
                                               ↓
                                      residual injection
```

本项目与原论文设置存在两个重要差异：我们在已有 Qwen3-0.6B-Base 上做短预算 continued pretraining，而不是从头预训练 recipient；教师使用 Qwen3-8B-Base，而论文教师对比中表现最好的是更强的 Qwen3.5-35B-A3B。因此本仓库检验的是一种低成本迁移场景，结果不能直接视为原论文复现。

### Controlled comparisons

| 组 | 训练内容 | Memory | 回答的问题 |
|---|---|---|---|
| B0 | 不训练 | 无 | 原始 Base 水平是多少？ |
| L | LoRA | 无 | 普通 continued pretraining 能提升多少？ |
| R | LoRA + adapter | 统计量匹配的随机表 | 额外参数与随机容量是否足够？ |
| S | LoRA + adapter | 打乱 key-row 对应的教师表 | 保留教师向量集合、破坏语义对应后会怎样？ |
| G | LoRA + adapter | 正确教师 n-gram 表 | 正确教师对应是否带来额外收益？ |

G−S 是最关键的语义检验。G 和 S 拥有相同的教师向量集合、参数量与命中率，区别只是 n-gram 能否取回与它对应的教师表示。G−L 则检验 memory 分支是否优于普通 LoRA。

<a id="key-finding-1"></a>

## 🧪 Key Finding 1：教师与学生表示可以对齐

我们先不训练下游模型，而是单独学习 Qwen3-8B T12 表示到 Qwen3-0.6B S1 表示的投影。60k 个 n-gram key 按 key 隔离为 train/validation/test；最终指标只在从未参与投影训练和方法选择的 held-out test keys 上计算。

| 映射 | Held-out cosine | Recall@1 | Recall@10 | CKA |
|---|---:|---:|---:|---:|
| Correct teacher mapping | 0.7821–0.7825 | 93.70%–94.04% | **99.56%–99.61%** | 0.899–0.901 |
| Shuffled teacher mapping | 0.3033–0.3045 | 0%–0.05% | 0.49%–0.59% | 0.058 |

简单 mean-centering 加 cosine+InfoNCE 优于 PCA-256、whitening-256 和 whitening+L2。结果表明教师 latent 中确实存在与对应学生 n-gram 表示相关、并能泛化到未见 key 的信息。它证明“表示可对齐”，但不保证下游选择题准确率提高。

🔗 [完整 alignment 设计与结果](docs/experiments/ALIGNMENT_STUDY.md)

<a id="key-finding-2"></a>

## 🧪 Key Finding 2：2M 信号没有通过 5M 确认

将 1024 维 aligned table 接回学生后，我们比较 frozen identity、low-LR identity、random projection，以及 G/S/L 三种语义对照。所有配置固定 T12→S1，并关闭 fallback 与 ShortConv。

### 2M-token result

CMMLU 学科 macro accuracy，三个 seed 的均值 ± 标准差：

| 配置 | CMMLU | 相对对照 |
|---|---:|---:|
| Low-LR aligned G | **51.100±0.029** | — |
| Low-LR shuffled S | 50.862±0.014 | G−S **+0.238 pp** |
| LoRA-only L | 50.870±0.019 | G−L **+0.229 pp** |
| Random-projection G | 50.951±0.118 | aligned G 高 +0.149 pp |

G−S 和 G−L 在 seeds 42/43/44 上均为正，配对 95% t 区间不跨 0。这是项目中控制最完整的正向信号，但增益只有约 0.23 个百分点，而且 C-Eval 没有同步提升。

### Pre-registered 5M confirmation

5M 实验没有续接 2M checkpoint，而是从同一 Base 权重独立训练，并让 cosine learning-rate schedule 覆盖完整 5M 预算。

| 配置 | CMMLU | C-Eval validation |
|---|---:|---:|
| Low-LR aligned G | 50.871±0.082 | 50.158±0.591 |
| Low-LR shuffled S | 50.946±0.118 | 50.203±0.208 |
| LoRA-only L | 50.902±0.101 | 50.354±0.509 |

CMMLU 的逐 seed G−S 为 `−0.163/+0.022/−0.085`，均值 `−0.075 pp`，95% t CI `[−0.306,+0.156]`；G−L 均值为 `−0.031 pp`，95% t CI `[−0.309,+0.247]`。67 个 CMMLU 学科中只有 3 个学科的 G−S 在三个 seed 上均为正，没有形成稳定受益领域。

**Interpretation：** 2M 的结果更像某一训练阶段出现的短暂小优势，而不是随训练继续扩大的能力增益。预注册确认失败后，不能再把 2M 结果表述为稳定提升。

🔗 [完整实验总结](docs/EXPERIMENT_RESULTS.md) · [原始 5M 结果](remote_results/aligned_5m/summary.json)

<a id="key-finding-3"></a>

## 🧪 Key Finding 3：模型使用了 memory，但收益停留在局部 loss

5M 训练完成后，我们在同一个 checkpoint 上分别正常开启 memory 和将 memory 输出置零，比较命中位置的 next-token loss。

| 诊断 | Correct G | Shuffled S |
|---|---:|---:|
| 最终 residual scale `alpha` | 约 0.004 | 接近 0 |
| 有效注入 / hidden norm | 1.49%–1.54% | 0.05%–0.11% |
| 关闭 memory 后的 hit-token loss | 三个 seed 均上升 `0.00018–0.00030` | 效果弱且不稳定 |

优化器能够区分正确表与打乱表，并主动采用正确 memory。因此 5M 失败不能解释为 gate 没打开或 memory 支路完全失效。更可能的情况是：memory 帮助了频繁局部短语的 next-token prediction，但作用量太小，或与 CMMLU/C-Eval 所需的知识检索和推理能力不匹配。

<a id="experiment-roadmap"></a>

## 🗺️ 实验路线

| 阶段 | 核心问题 | 主要结果 | 状态 |
|---|---|---|:---:|
| Pilot | 构表、训练、评测和 on/off 能否跑通？ | 完成端到端链路，未证明教师优势 | ✅ |
| Gate | 是否因为 gate 太小而学不到？ | memory 确实注入，差异仍不显著 | ✅ |
| Layer | 哪个教师层/学生层组合更好？ | 单点波动存在，24 项区间全部跨 0 | ✅ |
| Long | 2M 训练是否太短？ | 延长到 10M 仍无稳定 G−R/G−L 优势 | ✅ |
| Semantic | 教师语义是否优于打乱对应？ | 正确表未稳定超过 shuffled | ✅ |
| Mechanism repair | 扩表、fallback、ShortConv 是否改善？ | 单 seed C-Eval 上升，CMMLU 未复现 | ✅ |
| Confirmation | 单 seed 信号能否跨 seed？ | C-Eval 反转，CMMLU G−S −0.707 pp | ✅ |
| Alignment | 教师空间能否映射到学生空间？ | Held-out Recall@10 约 99.6% | ✅ |
| Aligned 2M | 对齐后是否产生下游收益？ | CMMLU G−S +0.238 pp | ✅ |
| Aligned 5M | 2M 信号能否独立确认？ | G−S −0.075 pp，确认失败 | ✅ |
| Temporal analysis | 2M 优势何时形成和消失？ | 待运行密集 checkpoint 分析 | ⬜ |
| Task-aware alignment | 对齐目标能否直接服务下游预测？ | 待实验 | ⬜ |

### Why each stage was needed

1. **Pilot** 先证明整个系统能工作，避免在实现错误上做大规模实验。
2. **Gate** 排除“残差系数从零附近开始，短训练没有打开”的解释。
3. **Layer** 检查教师表征抽象程度与学生插入深度是否需要匹配。
4. **Long** 检查早期无收益是否仅仅来自 token budget 不足。
5. **Semantic** 引入 shuffled teacher，将教师语义与额外参数、命中率和向量分布分离。
6. **Mechanism repair** 补入更接近论文的 500k 表、hash fallback 和 causal ShortConv。
7. **Confirmation** 用三个 seed 和组件关闭实验复核单 seed 的 C-Eval 积极结果。
8. **Alignment** 在下游训练前直接检验教师 latent 能否映射到学生 residual space。
9. **Aligned 2M** 用 frozen、low-LR 和 random projection 判断正向信号是否来自正确映射。
10. **Aligned 5M** 按预注册规则独立确认 2M 信号，最终确定其训练预算敏感性。

<a id="quick-start"></a>

## ⚙️ Quick Start

在已有 CUDA PyTorch 的 Python 环境中：

```bash
git clone https://github.com/tx1103mark/memory-grafting.git
cd memory-grafting
python -m pip install -r requirements.txt
python -m scripts.download_assets --root .
python -m scripts.prepare_data --root .
python -m pytest -q tests
```

模型与数据 revision 会写入 `assets.lock.json`。原始数据、模型权重、memory table 和 checkpoint 不提交到 Git。

构建 Qwen3-8B-Base block 12 的初版 memory：

```bash
export CUDA_VISIBLE_DEVICES=YOUR_ALLOCATED_GPU
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
python -m scripts.build_memory --root . --teacher-block 12
```

<a id="reproduce"></a>

## 🚀 Reproduce the Experiments

### Basic L/R/G comparison

```bash
python -m scripts.train --root . --group L --seed 42 --tokens 2000000
python -m scripts.train --root . --group R --seed 42 --tokens 2000000
python -m scripts.train --root . --group G --seed 42 --tokens 2000000
```

### Evaluation with lm-evaluation-harness

```bash
python -m scripts.evaluate --root . \
  --checkpoint runs/EXPERIMENT/last.pt \
  --tasks cmmlu,ceval-valid \
  --num-fewshot 0 \
  --output runs/EXPERIMENT/eval.json
```

评测使用 Base 模型普通文本补全，不应用 chat template。`ceval-valid` 是开发/辅助指标；最终配置判断优先使用 CMMLU、多 seed 对照和预注册确认。对于 G/S/R checkpoint，可增加 `--memory-mode zero` 在同一模型权重上关闭 memory。

### Full study runners

```bash
python -m scripts.run_layer_study
python -m scripts.run_semantic_study
python -m scripts.run_alignment_study
python -m scripts.run_aligned_downstream
python -m scripts.run_aligned_5m
```

这些 runner 面向本项目的服务器目录、数据和 GPU 布局。执行前应检查脚本配置、实际 GPU 空闲情况与磁盘空间；共享机器上不要终止其他用户的进程。

<a id="terminology"></a>

## 📚 Terminology

### What is held-out?

`Held-out n-gram` 指没有参与 alignment 投影训练和方法选择的 n-gram。60k keys 被划分为 train/validation/test：train 学习投影，validation 选择 mean-centering、PCA、whitening 和损失函数，test 只用于最终报告。它是离线 alignment 的测试集，不是 CMMLU/C-Eval，也不表示这些 key 被永久排除在最终 500k memory 之外。

### What is Recall@10?

教师向量投影到学生空间后，在学生表示库中按相似度检索。若正确 n-gram 对应的学生表示落在前 10 名，就记为命中。高 Recall@10 说明跨模型映射能保留 n-gram 身份，但不直接代表生成或答题准确率。

### What is low-LR?

Aligned memory 的 key/value 投影从 identity 初始化后继续训练，但使用较低学习率 `2e-6`。`frozen` 完全固定 identity；`random projection` 使用同一张正确教师表，但随机初始化投影。Low-LR 的目的，是允许学生轻微校准已对齐空间，同时减少短预算 continued pretraining 对离线映射的破坏。

### What are pp and 95% t CI?

`pp` 表示百分点，例如 51.100% − 50.862% = `+0.238 pp`。95% t CI 根据三个相同 seed 的配对差值计算；区间跨 0 表示这三次运行无法稳定判断 G 更好。三个 seed 仍然很少，因此不跨 0 的结果也需要更大预算独立确认。

### What is hit-token loss?

它只统计当前位置命中 n-gram memory 时，对下一个 token 的预测损失。对同一 checkpoint 比较 memory normal/zero，可以确认预测是否直接依赖 memory 支路。它是机制诊断指标，不等价于 CMMLU/C-Eval 能力。

## 🗂️ Repository Structure

```text
graft/
  data.py                 n-gram 查找与训练数据拼接
  model.py                Qwen3 显式前向与 memory adapter
  harness.py              lm-evaluation-harness 模型适配
scripts/
  prepare_data.py         数据清洗、去重和 token 化
  build_memory.py         教师 hidden-state 构表
  train.py                token-budgeted CLM + LoRA 训练
  evaluate.py             C-Eval/CMMLU 评测
  run_*_study.py          各阶段实验调度
  summarize_*.py          多 seed 汇总与配对统计
docs/
  EXPERIMENT_RESULTS.md   完整实验结果与解释
  TRAINING_PLAN.md        数据、训练和评测协议
  RUN_STATUS.md           历史运行记录
  experiments/            阶段预注册方案和报告
remote_results/           harness JSON 与诊断结果
```

<a id="documentation"></a>

## 📑 Documentation & Results

- [完整实验总结](docs/EXPERIMENT_RESULTS.md)
- [训练与评测方案](docs/TRAINING_PLAN.md)
- [阶段实验索引](docs/experiments/README.md)
- [显式 alignment 与 5M 确认](docs/experiments/ALIGNMENT_STUDY.md)
- [三 seed 机制确认](docs/experiments/CONFIRMATION_RESULTS.md)
- [历史运行记录](docs/RUN_STATUS.md)
- [原始与汇总评测结果](remote_results/)

## 🔭 Research Roadmap

- [x] Base-to-Base training and harness evaluation
- [x] Gate and injection-strength diagnostics
- [x] Teacher/student layer ablation
- [x] Random and shuffled semantic controls
- [x] 500k table, fallback and ShortConv ablation
- [x] Explicit teacher-to-student alignment
- [x] Three-seed 2M and pre-registered 5M evaluation
- [ ] Dense checkpoint analysis between 1M and 5M tokens
- [ ] Frequency- and hit-stratified benefit analysis
- [ ] Task-aware contrastive projection or online alignment loss
- [ ] Stronger grafting model such as Qwen3.5-35B-A3B

## 🙏 Acknowledgements

本项目受到以下工作的启发：

- [Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory](https://arxiv.org/abs/2605.20948)
- [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372)
- [TinyEngram](https://github.com/AutoArk/tinyengram)，本 README 的信息组织参考了其开放研究项目风格
- [Qwen3](https://github.com/QwenLM/Qwen3)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

本仓库保留所有负结果和对照实验。任何“提升”都应同时满足正确表优于 shuffled、优于 LoRA，并能在多 seed 和独立训练预算下复现。
