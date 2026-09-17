# 0.5M Memory + Engram fallback + ShortConv

本轮先用 60k 表进行 held-out alignment probe。T12→S1 correct/shuffled 的线性投影 cosine 为 0.813/0.493，CKA 为 0.00665/0.00212；T8→S12 为 0.909/0.664，CKA 为 0.1114/0.000015。教师表存在可学习对齐，T8→S12 尤其明显。

将 memory 扩展至 500,000 entries，2/3/4-gram 分别为 166,667/166,667/166,666，仍按训练文档频率选择、最低 df=2、tokenizer round-trip 有效。使用原 50M-token 去污染语料重新统计并生成 memory IDs。

hit 使用冻结教师表；miss 使用两个 65,536 buckets × 128 dim 的可训练 2/3-gram hash embedding，拼接后走独立 K/V 投影。选中的 K/V 经过 query-key sigmoid gate，value update 再经过 kernel=4 的零初始化因果 depthwise ShortConv，并残差写入学生。教师表冻结，fallback、投影、gate、ShortConv 与 LoRA 训练。

首轮仅 seed 42，T8→S12 与 T12→S1 各跑 G/S，外加共同 L；预算 5M tokens，保存 0.5M/1M/2M/5M 学习曲线。每个点评测 C-Eval validation，5M 评测 CMMLU和100k-token on/off诊断。主要比较 G−S；若仍无正向趋势，不直接扩展多 seed。
