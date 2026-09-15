# 教师来源层与 10M-token 训练实验

2026-09-14。参考 [Memory Grafting v1](https://arxiv.org/html/2605.20948v1)：§4.3.1 的来源层分析偏向第 6 层；附录 B.5 的单层实验是 Qwen3.5-35B-A3B Layer 6 注入学生第一层。A.1 双层配置为教师 [8,24]、学生 [1,12]。不能把第 6 层视为跨架构通用最优值；本文档 T/S 均明确指完成的 block 数（从 1 开始），不声称论文实现的索引约定与我们完全相同。

## 固定实验矩阵

| 配置 | 目的 | 组别 |
|---|---|---|
| T6 → S1 | 参考论文单层配置 | G / R |
| T4 → S6 | 前轮 G 分数最高的候选 | G / R |
| T8 → S12 | 前轮 G−R 差值较大的候选 | G / R |
| 无 memory | 同预算 LoRA 基线 | L |

第二台 H200 补充 T4→S1、T8→S1、T12→S1、T6→S6 的 G/R。它与主矩阵合并后，构成固定 S1 的 T4/T6/T8/T12 来源层比较，并检验 T6 在 S1 与 S6 的差异。补充矩阵由 `scripts/run_long_source_study.py` 执行，产物位于 `runs/long_source_study/`。

全部从 Qwen3-0.6B-Base 重新开始，教师 Qwen3-8B-Base；seed=42，10,000,000 有效预测目标 tokens，micro-batch=2、accumulation=2，alpha 初值 .001。同一数据顺序、60k keys、LoRA 与投影学习率沿用前轮，冻结教师表。5% token warmup + cosine 覆盖完整 10M；不续接已衰减完成的旧 2M checkpoint。

## 轨迹与评测

保存首次跨过 2M / 5M 的 optimizer-step checkpoint，以及准确 10M 的 last.pt。中途文件名表示阈值，实际 tokens 以 checkpoint 为准（没有截断或跳过中途 batch）。同预算所有组数据一致。2M 中途点使用 10M 日程，不能等同于前轮独立 2M 日程实验。

每个检查点运行 lm-evaluation-harness C-Eval validation、0-shot、likelihood、无聊天模板，报告样本加权准确率、学科宏平均、G−R/G−L 配对 bootstrap CI。10M 的 G 另做 memory-off 因果干预；G/R 记录 gate、实际注入比例和相同 100k 验证子集的 on/off loss。CMMLU 与 C-Eval test 保留，不用于此次选配置。

训练途中 loss 使用固定约 100k 验证子集，最终 loss 使用约 1M；二者不能直接连成同一验证曲线。使用指定 token checkpoint，不能根据不同验证子集混合挑选 best.pt。单 seed 的 CI 不反映训练随机性，也未校正多配置选择偏差。本实验检验本项目的 LoRA 继续预训练版本，不等于原论文从零预训练、Engram fallback、ShortConv 的完整复现。

## 运行与产物

Linux：`.venv/bin/python -u -m scripts.run_long_study`。203 主机仅使用 GPU 0–3，GPU 3 先构建 T6 表，随后四个 worker 消费七组训练队列；启动前每卡检查至少 20GB 空余。文件锁避免重复启动，不终止其他进程。错误写入 errors.json，修复后可用相同命令恢复。

后台提交：`bash scripts/launch_long_study.sh`；只读状态：`.venv/bin/python -m scripts.long_status`。本地训练/恢复测试通过（2 passed），覆盖两个学生位置，以及跨 token 阈值快照在完整训练与恢复训练之间的逐参数一致性。

`runs/long_*/` 保存 checkpoint / manifest / metrics；`runs/long_study/` 保存三阶段 harness 结果、逐题样本、诊断与最终 summary.json；`logs/long_study/` 保存日志。
