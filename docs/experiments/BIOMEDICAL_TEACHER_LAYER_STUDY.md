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

## Alignment 结果

所有层的正确映射都通过了 held-out key 检查。越浅的教师层越容易映射到学生 S1：

| 教师 block | Cosine | Recall@1 | Recall@10 | CKA |
|---|---:|---:|---:|---:|
| T4 | **0.9238** | **97.07%** | **100.00%** | **0.9650** |
| T6 | 0.9140 | 96.14% | 99.95% | 0.9601 |
| T8 | 0.8975 | 91.89% | 99.56% | 0.9492 |
| T12 | 0.8585 | 95.02% | 99.85% | 0.9212 |

这符合表征深度差异的直觉：教师浅层与学生浅层更相似。不过，下游结果没有随 alignment cosine 同步提高。

## Clinical Knowledge 主结果

数值为 seeds 42/43/44 的 accuracy 均值 ± 样本标准差：

| 教师 block | G | S | L | G−S | G−L |
|---|---:|---:|---:|---:|---:|
| T4 | 58.994±0.871 | 59.623±0.377 | 59.371±0.786 | **−0.629±0.576** | −0.377±0.998 |
| T6 | 59.245±0.377 | 59.245±0.000 | 59.371±0.786 | **0.000±0.377** | −0.126±0.950 |
| T8 | 59.119±0.786 | 59.623±0.377 | 59.371±0.786 | **−0.503±0.436** | −0.252±0.436 |
| T12 | **59.623±0.998** | 58.868±0.998 | 59.371±0.786 | **+0.755±0.654** | **+0.252±0.218** |

逐 seed G−S：

| 教师 block | seed 42 | seed 43 | seed 44 | 95% paired t CI |
|---|---:|---:|---:|---:|
| T4 | −1.132 | −0.755 | 0.000 | [−2.061, +0.803] |
| T6 | −0.377 | 0.000 | +0.377 | [−0.937, +0.937] |
| T8 | 0.000 | −0.755 | −0.755 | [−1.586, +0.579] |
| T12 | +0.377 | +0.377 | +1.509 | [−0.869, +2.378] |

四个区间都跨 0，因此当前题量无法证明任一层具有统计稳定的收益。但按预注册判据，T12 是唯一满足 G−S 三 seed 全正、G−L 三 seed 不为负的层；T4/T8 明确不应替换 T12，T6 与 shuffled 持平。

## 机制诊断

seed 42 上，T4/T8 正确表关闭后 hit-token loss 分别上升 `0.000272/0.000197`，说明它们也被模型使用；T6 正确表的变化接近 0。正确表的有效注入/hidden norm 为 T4 `0.61%`、T6 `1.17%`、T8 `0.86%`、T12 `1.29%`。注入强度本身不能解释层位排序。

最关键的观察是：T4 alignment 最好，Clinical 却最差；T12 alignment 最难，下游 G−S 反而最好。离线“能否重建学生浅层表示”不是充分的教师层选择标准。浅层教师向量可能主要重复词法和局部模式，而 T12 虽然与 S1 空间距离更远，却包含更多能被选择题利用的语义信息。

## 结论

1. 保留 T12→S1 作为当前 Biomedical 配置，不切换到论文中常见的 T6。
2. 教师层不能只按 cosine、CKA 或 Recall 选择；必须经过 G/S/L 下游语义对照。
3. 本次没有发现优于 T12 的教师来源层，且所有 95% 区间仍跨 0。结论是“在已测层中 T12 相对最好”，不是“已经证明 T12 普遍最优”。
4. 若继续层位研究，下一项更有判别力的实验是 T12 与更深层 T16/T20 的小规模比较，前提是扩大医学评测题量；继续密集扫描 T4–T12 的价值较低。

原始 harness、diagnostic、alignment 和汇总 JSON 位于 [`remote_results/biomedical_teacher_layers`](../../remote_results/biomedical_teacher_layers)。
