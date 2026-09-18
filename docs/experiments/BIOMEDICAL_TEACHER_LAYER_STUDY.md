# Biomedical Teacher Block Ablation

## 问题

在 Biomedical 领域设置中，教师 hidden state 的来源层是否影响 Memory Grafting 效果？本实验只改变 Qwen3-8B-Base 的教师 block，学生注入位置固定为 Qwen3-0.6B-Base block 1 后。

## 预注册配置

- 教师 blocks：T4、T6、T8、T12，编号表示完成的 Transformer block 数，从 1 开始。
- 学生：固定 S1。
- 每个教师层使用相同的 60k Biomedical 2/3/4-gram keys，分别重新抽取教师向量。
- 每个教师层独立学习 teacher-to-student alignment；投影选择只看 key-disjoint validation，最终 alignment 指标只看 held-out test keys。
- 下游训练：2M tokens，seeds 42/43/44，aligned low-LR，关闭 fallback 与 ShortConv。
- 语义对照：每层均运行正确表 G 和固定置换的 shuffled 表 S；LoRA-only L 在相同 seed 下共享。
- 评测框架：`lm-evaluation-harness==0.4.13`，Base 模型、0-shot、无 chat template。
- 主指标：`mmlu_clinical_knowledge` 的 G−S；G−L 为必要的第二判据。
- 辅助指标：Professional Medicine、Medical Genetics、Anatomy；World History 检查非领域能力保持性。

## 判读规则

1. 先确认每层 alignment 的正确映射明显优于 shuffled，再解释下游结果。
2. 以三 seed 配对 G−S 的均值排序教师层；G−L 用于排除收益只来自领域 continued pretraining。
3. 同时报告每个 seed、均值、标准差和配对 95% t 区间。仅凭单 seed 或单题差异不选择层位。
4. 若某层 G−S 三 seed 全正、G−L 不为负且优于 T12，才将其视为新的候选层；否则保留 T12 或判定层位差异尚不明确。
5. MMLU Clinical Knowledge 只有 265 道题，约 `0.377 pp` 对应一道题。即使方向一致，也必须按实际题数和区间保守解释。

## 复现

```bash
python -m scripts.run_biomedical_teacher_layer_study
```

脚本可断点恢复：已完成的构表、alignment、checkpoint 和评测不会重复执行。T12 复用已有三 seed 结果，避免重复消耗算力。
