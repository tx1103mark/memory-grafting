# 显式教师—学生 Alignment 实验

本阶段先做离线判别，不直接继续下游训练。固定 Qwen3-8B-Base block 12 教师表与 Qwen3-0.6B-Base block 1 学生目标，对 60k 个确定性抽样 key 做 70/15/15 key-disjoint 划分。PCA/whitening 只使用 train keys，方法和目标只看 validation，最终报告 held-out test。

第一阶段筛选 raw、center、remove-top-16、PCA-256、whiten-256、whiten-256+L2。每种方法分别训练正确表和固定 shuffled 表的同配方线性投影。第二阶段对 validation margin 最好的变换，使用三个 projection seed 比较 cosine 与 cosine+InfoNCE。

指标包括 held-out cosine、normalized MSE、CKA、Recall@1 和 Recall@10。通过标准是三个 projection seed 的正确表 test cosine 均高于 shuffled；下游实验前还要检查检索 margin，而不能只依赖 cosine。

通过后导出 1024 维 `aligned_table.pt`。下一阶段只测试 T12→S1，不启用 fallback/ShortConv，比较 frozen projection 与低学习率 projection，并以 CMMLU 为主指标。
