# 教师提取层 × 学生注入位置消融

状态：2026-09-14 21:10 全部完成，无失败任务。结果见 `LAYER_RESULTS.md`。新增层位检查通过；默认/非默认学生层号的精确断点恢复均通过。Bootstrap 文件与样本顺序已固定，避免归档后的文件顺序影响同 seed 区间。

层号一律从 1 开始：T4 表示教师第四个 block 的原始输出（无最终 norm）；S3 表示学生第三个 block 完成后加入 memory 残差。

矩阵：教师 T4/T8/T12 × 学生 S1/S3/S6/S12，每个组合均有教师表 G 和随机表 R，共 24 个记忆模型。对应教师层的随机表使用相同随机种子，按该层向量统计生成并匹配每行范数。各层使用同一套 60k n-gram key。构表仍只编码短片段，不额外引入原文上下文。

固定训练配置：Base→Base、seed 42、2M target tokens、micro-batch 2、accumulation 2、alpha 初值 0.001、gate bias -2、相同学习率和 token 日程。每组预计 747 次更新。已有 gate_G_a001、gate_R_a001 是 T12→S3，直接复用；gate_L 是共用 LoRA 基线，新增 22 次训练。

主要比较是每个组合 G−R，其次 G−L。每项用按学科配对 bootstrap 报告宏平均差及区间，保留题量加权 accuracy。只使用 C-Eval val 筛选；12 个组合的探索性选择不构成独立测试证据。单 seed 不覆盖训练随机性。

实现将 student_block、teacher_block、memory_dir 与 memory manifest 哈希存入新 checkpoint。评测恢复对应层号和记忆表，旧 checkpoint 默认 T12→S3。不同位置训练必须重新训练，不能只在推理时移动同一 adapter。

入口：`scripts/run_layer_study.py`。后台采用每 GPU 一个 worker 的任务队列，单卡可用显存至少 20GB 才启动。用户授权共享全部 H200 GPU；不终止其他进程。完成训练后自动执行 lm-evaluation-harness，最后生成 `runs/layer_study/summary.json` 和 `complete`。

查看进度：在服务器项目目录执行 `.venv/bin/python -m scripts.layer_status`；主日志为 `logs/layer_study_driver.log`。失败后同入口可复用已完成结果，已有训练 checkpoint 按相同配置显式恢复。
