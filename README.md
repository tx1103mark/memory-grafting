# Qwen3 Base Memory Grafting

本项目研究一个具体问题：能否把 **Qwen3-8B-Base 的离线 n-gram hidden state** 作为冻结条件记忆，注入 **Qwen3-0.6B-Base**，通过少量 continued pretraining 提升小模型的中文能力。

项目已经完成从端到端 pilot、层位与门控消融、三 seed 语义对照、机制修复、显式教师—学生 alignment，到 5M-token 预注册确认的完整实验链路。评测统一使用 `lm-evaluation-harness==0.4.13`，主任务为 CMMLU，辅助任务为 C-Eval validation。

## 当前结论

**Memory Grafting 支路可以被学生模型使用，但当前训练设置没有产生稳定的 CMMLU/C-Eval 能力提升。**

- 正确教师表确实包含可迁移结构。显式 alignment 后，held-out n-gram 的 Recall@10 为 **99.56%–99.61%**，打乱表仅为 **0.49%–0.59%**。
- 2M-token 下游实验出现过小而一致的 CMMLU 信号：low-LR G 相比 shuffled S 为 **+0.238 pp**，相比 LoRA-only L 为 **+0.229 pp**，三个 seed 均为正。
- 预注册的 5M-token 独立训练没有复制该信号：G−S 为 **−0.075 pp**，G−L 为 **−0.031 pp**，95% t 区间均跨 0；C-Eval 同样没有提升。
- 5M memory on/off 诊断显示，关闭正确表后三个 seed 的 hit-token loss 均上升 `0.00018–0.00030`，而 shuffled 表的注入强度趋近于零。这说明模型能识别并使用正确 memory，但局部 next-token loss 收益没有稳定转化为中文多选知识准确率。

因此，现阶段最稳妥的判断是：**显式 alignment 能产生可检测但训练预算敏感的弱信号，尚不能证明 Memory Grafting 稳定提高现有 Qwen3-0.6B-Base 的中文能力。** 继续单纯增加相同 causal-training tokens 或扩大表规模的判别价值有限。

### 如何理解这些结论

**Held-out n-gram** 是没有参与 alignment 投影训练和超参数选择的 n-gram。我们把 60k 个 n-gram key 按 key 隔离为 train/validation/test：train 用于学习教师空间到学生空间的线性投影，validation 用于选择 mean-centering、PCA、whitening 及训练目标，最后只在 held-out test keys 上报告结果。它相当于离线对齐实验的测试集，用来检查投影能否泛化到未见过的短语，而不是记住训练表项。这里的 held-out 不是 CMMLU/C-Eval，也不是从最终 500k memory 中永久删除这些 key。

**Recall@10** 衡量一个教师表向量经过投影后，能否在学生表示库中找回与它属于同一 n-gram 的正确表示。`99.56%–99.61%` 表示约 99.6% 的 held-out 查询，其正确学生表示位于相似度最高的 10 个候选中。打乱 key-row 对应后只有约 0.5%，说明正确教师表示与对应学生表示之间确实存在可泛化映射。但这只是“表示可对齐”的证据，不等于下游答题能力会提高。

**Low-LR** 指 aligned memory 的 key/value 投影从 identity 初始化后仍允许训练，但使用较低的学习率 `2e-6`。作为比较，`frozen` 固定 identity 投影不更新，`random projection` 使用同一张正确教师表却随机初始化投影。Low-LR 允许模型轻微校准已经对齐的空间，同时尽量避免短预算 continued pretraining 破坏离线学到的映射。

**G、S、R、L** 分别表示正确教师表、打乱教师对应关系的表、随机向量表和 LoRA-only。G−S 是最关键的语义检验：两组拥有完全相同的教师向量集合、参数量和查表命中率，区别只有 n-gram 是否取回正确的教师向量。G−L 则回答增加 memory 分支是否优于普通 LoRA continued pretraining。

**pp** 是百分点。例如 51.100% 减去 50.862% 等于 `+0.238 pp`。**95% t CI** 是根据三个相同 seed 的配对差值计算的置信区间；区间跨 0 表示现有三次运行无法稳定判定 G 更好。三个 seed 样本仍很少，因此即使区间不跨 0，也只能视为需要更大预算独立确认的信号。

**Hit-token loss** 只统计当前位置命中 n-gram memory 时，对下一个 token 的预测损失。对同一个训练完成的 checkpoint，分别正常开启 memory 和把 memory 输出置零，能够观察该支路对预测的直接贡献。关闭正确表后 loss 小幅上升，说明模型确实使用了 memory；但上升只有约 `0.00018–0.00030`，而 CMMLU/C-Eval 是离散的多选准确率，因此这种局部收益可能太小，或者与学科答题所需能力不一致。

2M 与 5M 并不矛盾。2M 时正确表相对 S/L 出现约 `+0.23 pp`，但效果很小；5M 独立实验改变了完整学习率日程，并从同一个 Base checkpoint 重新训练，结果回到零附近。这说明 2M 结果更像某一训练阶段的短暂优势，而不是随着训练增加会持续扩大的能力增益。

## 方法

对训练语料中的高频 2/3/4-gram，用教师模型离线编码最后一个 token 的 hidden state，构建确定性查找表。学生在指定 block 后检索最长匹配 n-gram，并通过投影、query-key gate 和残差连接注入：

```text
tokens ── exact longest-match lookup ── frozen teacher table
   │                                           │
   └── Qwen3-0.6B block l ── query-key gate ──┤
                                               ↓
                                      residual injection
```

实验中使用以下对照：

| 组 | 训练内容 | Memory |
|---|---|---|
| B0 | 不训练 | 无 |
| L | LoRA | 无 |
| R | LoRA + adapter | 与教师表统计量匹配的随机表 |
| S | LoRA + adapter | 保留教师向量集合、打乱 key-row 对应 |
| G | LoRA + adapter | 正确教师 n-gram 表 |

G 与 S 的差异用于检验教师语义对应关系；G 与 R 区分教师表示和随机容量；G 与 L 判断 memory 分支相对普通 continued pretraining 是否有实际收益。

## 数据与评测

- 学生：`Qwen/Qwen3-0.6B-Base`
- 教师：`Qwen/Qwen3-8B-Base`，主要候选为 teacher block 12
- 数据：Ultra-FineWeb 中文分片，无人工标签；输入经 Qwen3 tokenizer 转为 token IDs，训练标签为右移一位的 next token
- 清洗：中文比例与长度过滤、精确去重、MinHash 近似去重，以及 C-Eval/CMMLU 题干和选项的字符串重叠排除
- 初版表：60k 个 2/3/4-gram；机制修复与 alignment 阶段：500k entries
- 评测：Base 模型、0-shot、无 chat template；CMMLU 为确认指标，C-Eval validation 作为辅助指标
- 汇总：报告学科 macro accuracy；完整 harness 结果保存在 `remote_results/`

本项目是在已有 Base checkpoint 上做短预算 continued pretraining。Memory Grafting 原论文主要从头预训练 recipient，并使用更强的 Qwen3.5-35B-A3B 等 grafting model，因此两者结果不能直接等同。

## 实验过程

| 阶段 | 设置 | 结果与作用 |
|---|---|---|
| Pilot | 60k 表，B0/L/R/G，2M tokens | 打通构表、训练、memory on/off 与 harness 评测；未证明教师优势 |
| Gate | 不同 alpha/gate 强度，2M | 确认 memory 分支没有因初始化而关闭，六项差异区间均跨 0 |
| Layer | T4/T8/T12 × S1/S3/S6/S12，2M | T8→S12 单点 G−R 最大为 +0.729 pp，但所有配对区间均跨 0 |
| Long | 代表层位，2M/5M/10M | 延长预算仍未形成稳定 G−R/G−L 优势 |
| Semantic | T12→S1、T8→S12，G/S/R/L × 3 seeds，10M | T12→S1 CMMLU 的 G−S 为 −0.671 pp，否定初版语义迁移 |
| Mechanism repair | 500k 表 + hash fallback + ShortConv，0.5M–5M | 单 seed T12→S1 在 C-Eval 出现 +1.03 pp G−S，但 CMMLU 未复现 |
| Confirmation | 修复机制 G/S/L × 3 seeds | CMMLU G−S 为 −0.707 pp；关闭 fallback/ShortConv 反而略好 |
| Alignment | mean-centering + cosine/InfoNCE 投影 | held-out correct Recall@10 约 99.6%，证明教师和学生空间可对齐 |
| Aligned 2M | frozen/low-LR/random projection 对照 × 3 seeds | low-LR G−S +0.238 pp、G−L +0.229 pp，形成小幅 CMMLU 信号 |
| Aligned 5M | low-LR G/S/L × 3 seeds，预注册确认 | G−S −0.075 pp、G−L −0.031 pp，未复制 2M 信号 |

### 各阶段为什么这样推进

1. **Pilot：先验证整条链路。** 初版使用 60k 个高频 2/3/4-gram，把 Qwen3-8B-Base 第 12 个 block 的表示注入学生。B0、L、R、G 的训练、评测和 memory on/off 都能运行，证明实现具备继续实验的基础；但单次分数不足以判断教师知识是否有效。

2. **Gate：排除 memory 没有真正进入模型。** 我们改变 residual scale 的初值并记录 gate 和注入向量相对 hidden state 的范数。memory 分支实际处于工作状态，注入约占 hidden norm 的 1% 左右，但不同 gate 配置的评测区间都跨 0。因此早期失败不能简单归因于“gate 从零开始、训练太短而没有打开”。

3. **Layer：搜索教师来源层和学生插入层。** 比较教师 T4/T8/T12 与学生 S1/S3/S6/S12，希望找到表征抽象程度更匹配的位置。T8→S12 出现最大的单点 G−R 差值，但 24 项配对区间全部跨 0。这轮只能产生候选层位，不能确认最佳组合。

4. **Long：检查训练预算是否不足。** 对代表性层位训练到 10M tokens，并保留 2M/5M checkpoint。更长训练没有让正确表稳定超过随机表或 LoRA，说明仅延长相同训练目标不足以解决问题。

5. **Semantic：用 shuffled teacher 做严格语义对照。** 随机表 R 同时改变向量内容和对应关系，无法单独回答教师语义是否有用；因此加入 S，在不改变教师向量集合的情况下打乱 n-gram 与向量的对应。T12→S1 和 T8→S12 各跑 G/S/R/L × 3 seeds。正确表没有稳定超过 S，T12→S1 的 CMMLU G−S 反而为 −0.671 pp，否定了初版结构的语义迁移解释。

6. **Mechanism repair：补齐论文中的关键结构。** 将精确表从 60k 扩至 500k，未命中位置加入可训练 hash n-gram fallback，并增加因果 ShortConv。T12→S1 在单个 seed 的 C-Eval 上从 2M 到 5M 持续优于 S/L，但 CMMLU 没有同步提升。这说明更完整机制可能产生信号，也可能只是开发集和单 seed 波动。

7. **Confirmation：复核单 seed 积极结果并做组件归因。** 固定机制后运行三个 seed，同时分别关闭 fallback、ShortConv 和教师表。C-Eval 的优势在 seed 44 反转，CMMLU G−S 为 −0.707 pp；关闭 fallback 或 ShortConv 反而略好。因此上一阶段的 C-Eval 提升不能作为稳定结论，研究重点转向教师与学生表示空间不匹配。

8. **Alignment：先离线验证教师表示能否映射到学生空间。** 对正确表和 shuffled 表分别训练相同配方的投影，严格使用 train/validation/held-out test 划分。简单 mean-centering 加 cosine+InfoNCE 优于 PCA/whitening；正确表的 held-out cosine、CKA 和检索率远高于 shuffled。这证明教师 latent 里存在与学生 n-gram 表示对应的信息。

9. **Aligned 2M：把离线对齐表接回学生。** 比较 frozen、low-LR、random projection、G/S/L。Low-LR G 在三个 seed 上都超过 S 和 L，CMMLU 分别提升 +0.238/+0.229 pp；同一张表配随机投影也更差。这是整个项目中控制最完整的正向信号，但幅度很小，且 C-Eval 没有提升。

10. **Aligned 5M：按预注册标准做独立确认。** 固定 T12→S1、500k 表、关闭 fallback/ShortConv，只比较 low-LR G/S/L，三个 seed 全部从 Base 权重独立训练到 5M。CMMLU 和 C-Eval 均未复制 2M 优势。与此同时，on/off 诊断仍显示 G 被模型使用，所以最终问题不是支路失效，而是其局部语言建模收益没有形成稳定下游能力收益。

### 最终 5M 结果

分数为三个 seed 的学科 macro accuracy 均值 ± 标准差。

| 配置 | CMMLU | C-Eval validation |
|---|---:|---:|
| Low-LR aligned G | 50.871±0.082 | 50.158±0.591 |
| Low-LR shuffled S | 50.946±0.118 | 50.203±0.208 |
| LoRA-only L | 50.902±0.101 | 50.354±0.509 |

CMMLU 的逐 seed G−S 为 `−0.163/+0.022/−0.085`，95% t CI 为 `[−0.306,+0.156]`；G−L 为 `+0.098/−0.096/−0.096`，95% t CI 为 `[−0.309,+0.247]`。67 个 CMMLU 学科中只有 3 个学科的 G−S 在三个 seed 上均为正，没有形成稳定受益领域。

## 代码结构

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
  TRAINING_PLAN.md        训练设计、数据和评测协议
  RUN_STATUS.md           历史运行记录
  experiments/            各阶段预注册方案和报告
remote_results/           可复核的 harness JSON 与诊断结果
```

模型权重、原始数据、冻结 memory table、训练 checkpoint 和逐题 sample 文件不进入 Git。仓库保留生成代码、配置 manifest、汇总 JSON 和结论文档。

## 安装与准备

需要已有 CUDA PyTorch 的 Python 环境：

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets --root .
python -m scripts.prepare_data --root .
python -m pytest -q tests
```

模型与数据 revision 写入 `assets.lock.json`。构建初版教师表：

```bash
export CUDA_VISIBLE_DEVICES=YOUR_ALLOCATED_GPU
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
python -m scripts.build_memory --root . --teacher-block 12
```

基础 L/R/G 训练：

```bash
python -m scripts.train --root . --group L --seed 42 --tokens 2000000
python -m scripts.train --root . --group R --seed 42 --tokens 2000000
python -m scripts.train --root . --group G --seed 42 --tokens 2000000
```

训练可用 `--resume <checkpoint>` 精确恢复，但 token budget、数据、模型和 memory manifest 必须一致。服务器是共享资源；运行调度脚本前应重新检查 GPU 和磁盘，不应自动终止其他用户进程。

## 评测

```bash
python -m scripts.evaluate --root . \
  --checkpoint runs/EXPERIMENT/last.pt \
  --tasks cmmlu,ceval-valid \
  --output runs/EXPERIMENT/eval.json
```

评测固定 `--num-fewshot 0`，不使用聊天模板。`ceval-valid` 是开发/辅助指标；配置选择后的结论优先由 CMMLU 和多 seed 对照支持。G/S/R checkpoint 可通过 `--memory-mode zero` 在同一权重上关闭 memory，确认预测或 loss 是否实际依赖该支路。

## 文档与结果索引

- [完整实验总结](docs/EXPERIMENT_RESULTS.md)
- [训练与评测方案](docs/TRAINING_PLAN.md)
- [历史运行记录](docs/RUN_STATUS.md)
- [阶段实验文档](docs/experiments/)
- [显式 alignment 与 5M 确认](docs/experiments/ALIGNMENT_STUDY.md)
- [三 seed 机制确认](docs/experiments/CONFIRMATION_RESULTS.md)
- [原始与汇总评测结果](remote_results/)

## 后续方向

下一轮应优先回答“2M 的短暂增益为何在 5M 消失”，而不是继续增加相同训练预算：

1. 在 1M–5M 之间保存更密集 checkpoint，观察 G−S、alpha、注入范数和 hit-token loss 的时间动态。
2. 按 n-gram 频次、命中位置与 CMMLU 学科拆分收益，判断 memory 是否只帮助局部语言模式。
3. 将 alignment 目标加入训练过程，测试 task-aware contrastive projection 或显式 hidden-state alignment。
4. 只有当正确表在多 seed 下稳定优于 shuffled 和 LoRA，才扩展教师规模、memory 容量或扫描更多层位。
