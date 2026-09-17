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
