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
