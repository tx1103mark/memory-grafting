# 教师语义对齐确认实验

固定 T12→S1 与 T8→S12，比较 G（原教师表）、S（固定打乱教师 key-row 对应）、R（矩与行范数匹配的高斯随机表）、L（无 memory）。seed 为 42/43/44，每组从 Qwen3-0.6B-Base 训练 10M target tokens；复用已经完成的 seed 42 G/R/L checkpoint，只新增 seed 42 的 S。

S 使用种子 20260915 对表的非零行做 `randperm`。它保留教师向量的精确多重集合、范数、协方差、有效秩和各向异性，只破坏 n-gram key 与教师 latent 的语义对应。主要因果对比为 G−S；G−R 检验教师表相对随机 n-gram ID 特征，G−L 衡量整个 graft 方案。

每个最终 checkpoint 用 lm-evaluation-harness 做 C-Eval validation 和 CMMLU 0-shot likelihood 评测，并在固定 100k-token 验证集做 memory on/off loss、gate 和残差尺度诊断。配置在实验开始前锁定，不根据 CMMLU 调参。跨 seed 报告均值、标准差和配对 seed 差；题目 bootstrap 作为题集不确定性补充，不能替代训练 seed 方差。

203 主机运行 seed 43 七组、seed 42 的 T8→S12 S 及相关已有 checkpoint 评测；238 主机运行 seed 44 七组、seed 42 的 T12→S1 S 及相关已有 checkpoint 评测。完成标记为 `runs/semantic_study/complete-203` 与 `complete-238`。
