# Biomedical Domain Memory Grafting Pilot

## 目的

参考 TinyEngram 的领域适配设置，检验 Memory Grafting 在训练语料与评测任务领域一致时是否比通用中文 continued pretraining 更容易产生收益。主指标为 `mmlu_clinical_knowledge` test accuracy。

## 数据边界

TinyEngram 使用 Biomed-Enriched 训练，并使用 MMLU 医学子任务评测。本实验不使用 MMLU test 题训练：

- 训练来源：`almanach/Biomed-Enriched` 的 `commercial-00000-of-00026.parquet`，固定 revision `03a84da5ae6d75bd784b678972e46d880967f8e8`。
- 训练过滤：`language=en`、`domain=biomedical`、`educational_score>4.0`。
- 文本验证：相同分片中 `educational_score<4.0` 的样本。
- 去污染：排除与 MMLU Clinical Knowledge dev/validation/test 存在规范化 30 字符精确重叠的训练文本。
- 训练预算：5M 原始 train tokens；pilot 中每组消费 2M target tokens。
- Memory keys：训练文档频次最高且 tokenizer round-trip 有效的 2/3/4-gram，每阶 20k，共 60k。

MMLU Clinical Knowledge 本身只用于评测。直接训练其 test split 会造成数据泄漏；dev/validation 样本也仅用于污染排除，不用于优化。

## 配置

- 学生：Qwen3-0.6B-Base。
- 教师：Qwen3-8B-Base block 12。
- 注入：学生 block 1 后；领域 keys 上重新学习 teacher-to-student alignment。
- G/S 使用 aligned table、identity 初始化 K/V 与 `2e-6` low-LR；关闭 fallback 和 ShortConv。
- 组别：B0、LoRA-only L、正确教师表 G、打乱对应的教师表 S。
- pilot seed：42。若 Clinical Knowledge 的 G−S 与 G−L 同时为正，再运行 seeds 43/44。

## 评测

- 主指标：`mmlu_clinical_knowledge`。
- 医学辅助：`mmlu_professional_medicine`、`mmlu_medical_genetics`、`mmlu_anatomy`。
- 非医学回归：`mmlu_high_school_world_history`。
- 框架：`lm-evaluation-harness==0.4.13`，0-shot，不使用 chat template。

本实验是领域方向 pilot。单 seed 的正向结果只用于决定是否扩大到三个 seed，不能作为最终结论。

## 数据产物

- train：18,651 篇文档，5,000,137 tokens。
- validation：5,067 篇文档，500,008 tokens。
- 去污染命中：25 篇候选训练文本被排除。
- Memory：60,000 个 2/3/4-gram keys。
- 领域 alignment：held-out cosine `0.8585`、Recall@1 `95.02%`、Recall@10 `99.85%`、CKA `0.9212`。

这里的 held-out keys 没有参与投影训练或映射选择。高检索指标说明领域教师向量能够可靠映射到学生空间，但不直接等价于 MMLU 准确率。

## Pilot 结果

seed 42 达到预注册扩展条件：Clinical Knowledge 上 G 同时高于 S 和 L，因此继续运行 seeds 43/44。

| 组别 | Clinical | Professional Medicine | Medical Genetics | Anatomy | World History |
|---|---:|---:|---:|---:|---:|
| B0 | 58.868 | 51.471 | 57.000 | 47.407 | 62.025 |
| L | 60.000 | 49.265 | 53.000 | 47.407 | 62.025 |
| G | **60.377** | 50.000 | 54.000 | 45.926 | 60.759 |
| S | 60.000 | 50.368 | 54.000 | 45.185 | 60.759 |

Clinical 的 G−S 与 G−L 均为 `+0.377 pp`，即 265 道题中的 1 题。该差值足以按预注册规则进入多 seed 确认，但本身不是有统计把握的提升。

## 三 seed 确认

下表为 seeds 42/43/44 的均值 ± 样本标准差：

| 组别 | Clinical | Professional Medicine | Medical Genetics | Anatomy | World History |
|---|---:|---:|---:|---:|---:|
| L | 59.371±0.786 | 49.020±0.425 | 53.667±2.082 | 46.173±2.804 | 61.463±0.487 |
| G | **59.623±0.998** | **49.265±0.735** | **55.000±1.732** | **46.667±4.124** | **62.025±1.266** |
| S | 58.868±0.998 | 49.020±1.182 | 54.000±2.000 | 45.185±2.222 | 61.322±0.645 |

Clinical 的配对差值：

| 对照 | seed 42 | seed 43 | seed 44 | 均值 ± SD | 正向 seed |
|---|---:|---:|---:|---:|---:|
| G−S | +0.377 | +0.377 | +1.509 | **+0.755±0.654** | 3/3 |
| G−L | +0.377 | 0.000 | +0.377 | **+0.252±0.218** | 2/3，另 1 个持平 |

G−S 在三个 seed 上均为正，说明正确领域 key-row 对应关系比打乱教师表更好；G−L 只表现为两正一平，幅度约为不到 1 道题。按三个 seed 的配对 t 区间，两个差值都仍跨 0，因此证据应描述为“小而方向一致的语义信号”，而不能描述为已经证明稳定提升。

相对于固定 B0，G 的三 seed 均值高 `+0.755 pp`；其中普通领域 continued pretraining（L）已经贡献 `+0.503 pp`，教师表在 L 之上的均值增量为 `+0.252 pp`。这说明领域匹配数据本身是主要贡献者，memory 提供了更小的附加信号。

## 机制诊断

在 100k validation tokens 上，领域 n-gram 命中率约为 47.8%。正确表 G 的有效注入/hidden norm 在三个 seed 上为 `0.75%–1.29%`，打乱表 S 为 `0.01%–0.42%`。关闭 memory 后，G 的 hit-token loss 分别上升 `0.000062/0.000290/0.000326`；S 的变化更小且不稳定。

模型因此确实更愿意采用正确领域表。与通用中文 5M 实验相比，这次领域训练和领域评测匹配后，G−S 从不稳定或负向变成 Clinical 3/3 正向；不过增益仍只有 1–4 道题，样本量和 seed 数都不足以支持强结论。

## 当前结论与后续判据

1. 用 Biomed-Enriched 训练、MMLU Clinical Knowledge 评测的无泄漏流程已经跑通。
2. 正确医学教师表在主指标上三 seed 均优于 shuffled，平均 `+0.755 pp`；这是领域 memory 的初步正向证据。
3. 正确表相对 LoRA-only 仅 `+0.252 pp`，且有一个 seed 持平。大部分收益来自领域 continued pretraining。
4. 现阶段更适合扩大独立评测题量，或换用更大的医学 benchmark，而不是继续在同一 265 题集合上扫描超参数。
5. 若继续训练，应预注册更长预算与独立 test，并同时报告 G−S、G−L、逐题配对差异和通用能力保持性。

后续教师层消融比较了 T4/T6/T8/T12。虽然 T4 的 held-out alignment cosine 最高（0.924），只有 T12 在 Clinical 上取得三 seed 全正的 G−S；T4/T6/T8 的平均 G−S 分别为 `−0.629/0.000/−0.503 pp`。因此当前配置继续使用 T12，详见 [Biomedical 教师层消融](BIOMEDICAL_TEACHER_LAYER_STUDY.md)。

原始 harness JSON、诊断和汇总见 [`remote_results/biomedical`](../../remote_results/biomedical)。
