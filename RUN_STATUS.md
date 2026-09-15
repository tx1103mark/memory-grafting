# 实验执行状态

2026-09-15 17:16：语义确认实验全部完成，203/238 的错误列表均为空，共 21 个唯一 checkpoint 的 C-Eval 与 CMMLU 结果、18 份 memory 诊断。跨 seed 汇总显示 T12→S1 的 G−S 在 C-Eval 为 +0.014 pp、CMMLU 为 −0.671 pp；T8→S12 两任务均无法区分 G/S/R/L。当前简化接口没有稳定教师语义迁移证据，详见 `SEMANTIC_RESULTS.md`。

2026-09-15 后续：语义对齐确认实验已启动。固定 T12→S1 与 T8→S12，G/S(shuffled teacher)/R/L，seeds 42/43/44，10M tokens；两台 H200 共新增 16 组训练，并为全部最终 checkpoint 运行 C-Eval validation、CMMLU 和 memory 诊断。本地完整测试 9 passed，服务器核心测试各 7 passed。完成标记为 `runs/semantic_study/complete-203` 与 `complete-238`。

2026-09-15 09:18：双机 15 组 10M-token 训练、2M/5M/10M C-Eval validation、10M memory-off 与诊断全部完成。修复 `compare_results.py` 的精确 task 样本选择后，两边汇总完成；旧 glob 会把 `.off` 样本误并入 on 结果并触发重复 ID，但不影响原始评测。总体未证明教师表优于随机表；详见 `LONG_RESULTS.md`。

2026-09-14 21:46：10M-token 双机实验已实际启动。h200-203 使用 GPU 0–3 运行 L、T6S1/T4S6/T8S12 的 G/R 队列，首批四组已记录约 0.7M tokens；原 h200 运行 T4S1/T8S1/T12S1/T6S6 的 G/R，八组均已记录训练指标，其中四组已生成 2M 阈值快照（实际 optimizer 边界为 2,000,689 tokens，manifest 总日程为 10M）。T12 初次因目录写错立即失败，已修正为 `memory/` 并在 GPU 4/5 成功补启动；recovery watcher PID 979424 会在现有训练结束后用修正版 runner 补评测和汇总。两机训练均为后台任务，完成标记分别为 `runs/long_study/complete` 与 `runs/long_source_study/complete`。

2026-09-14 后续：已核对 Memory Grafting 附录 B.5 的教师第 6 层 → 学生第一层，新增 `LONG_STUDY.md` 和 `scripts/run_long_study.py`，准备 7 组 10M-token 实验（L，T6S1/T4S6/T8S12 的 G/R）。训练器新增 2M/5M 边界快照，本地真实小模型训练及精确恢复测试 2 passed（94.17s）。H200 的 SSH 多次在 banner exchange 阶段超时，尚未确认新训练启动；不要把代码已上传当作训练已运行。完成状态以 `runs/long_study/complete` 为准。

2026-09-14 21:10：教师层 T4/T8/T12 × 学生位置 S1/S3/S6/S12 的 G/R 交叉消融全部完成；22 组新增训练，2 组复用。每组 2M tokens、747 步。最高 G accuracy 为 T4→S6 的 51.3373%；G−R 宏平均差最大为 T8→S12 的 +0.7291 pp。24 项 G−R/G−L 的配对 95% 区间全部跨 0。完整结果见 `LAYER_RESULTS.md`，设计见 `LAYER_STUDY.md`；原始结果已同步到本地 `remote_results/runs/layer_study`，压缩包与服务器 SHA256 一致。

后续更新：Gate 对照的七组 2M-token 训练与 C-Eval 已完成；每组 747 次更新，alpha 初值为 0.001/0.01/0.05。六项 G−L/G−R 配对区间均跨 0。实测旧 G gate 均值约 0.804，实际注入约 1.37%，修正了此前 gate 未打开的猜测。详细结果见 `GATE_STUDY.md`。

更新时间：2026-09-14 19:47，Asia/Shanghai。此文件是时间点快照，实时状态以服务器日志和完成标记为准。

## 已完成

- 实现 Base → Base 的 L/R/G 共用训练器、冻结教师/随机记忆、最长后缀查表、LoRA、门控注入、token 预算控制、checkpoint 与精确断点恢复。
- 评测统一为 `lm_eval==0.4.13`，继承 HFLM 的 likelihood 评分逻辑。官方模板、描述和指标保留，数据改为固定 revision 的本地副本。
- H200 环境安装完成：PyTorch 2.10.0+cu128、Transformers 4.57.6、PEFT 0.18.0、Datasets 3.6.0、lm_eval 0.4.13。
- 服务器 CPU 测试：**8 passed**。覆盖原生前向一致性、因果性、查表/文档边界、padding/EOS 标签、梯度、保存重载、harness 多 token 评分一致性、训练入口及精确恢复。
- 数据准备完成：训练 **50,016,041 tokens / 49,757 documents**，验证 **1,002,390 tokens / 881 documents**；按文档划分，得到 2/3/4-gram 各 20,000 个 key。
- 数据过滤统计：质量过滤 5,701 篇、C-Eval/CMMLU 规范化连续 30 字符重叠过滤 73 篇、精确重复 9 篇、MinHash 近重复 11 篇。
- 学生、教师前 12 blocks 所需两个分片、中文语料均完成下载并通过固定 revision 对应的 SHA256 校验，`assets.complete` 已生成。
- Qwen3-8B-Base 前 12 blocks 已编码 60,000 个 key，教师记忆表形状为 `[60001, 4096]`；同分布随机对照表同步生成。
- Qwen3-0.6B-Base 零训练基线 B0 已在 C-Eval val 的 1,346 道题上完成：size-weighted accuracy **50.97%**，subject macro **49.75%**。
- L/R/G 各 2,000,000 target-token pilot 和 C-Eval val 评测均已完成，`runs/pilot.complete` 已生成。

## Pilot 结果

服务器 `h200`，目录 `/mnt/disk0/home/tysearch/ngram-embedding`：

| 组别 | C-Eval 加权 accuracy | 学科宏平均 | 最终验证 loss |
|---|---:|---:|---:|
| B0 | 50.9658% | 49.7481% | — |
| L | 50.6686% | 49.5247% | 2.9145763 |
| R | 50.2972% | 49.0841% | 2.9145052 |
| G | 50.5201% | 49.2719% | 2.9145021 |

- G−L 学科宏平均差为 **-0.2528 pp**，95% 配对 bootstrap CI `[-1.0361, 0.5303] pp`。
- G−R 学科宏平均差为 **+0.1877 pp**，95% 配对 bootstrap CI `[-0.6413, 1.0039] pp`。
- 当前单 seed、2M-token pilot 没有证明 Memory Grafting 有效。教师表优于随机表的方向很弱且不显著，G 仍低于 L 和 B0。
- 完整解释见 `PILOT_RESULTS.md`。

没有停止或修改服务器上其他使用者的进程；训练按用户授权共享 H200 GPU。

实时检查：

```bash
cd /mnt/disk0/home/tysearch/ngram-embedding
tail -n 20 logs/train_L.log
tail -n 20 logs/train_R.log
tail -n 20 logs/train_G.log
tail -n 20 logs/pilot_orchestrator.log
cat runs/pilot.complete
```

## 下载实现

下载流程吸收了 `H200_SERVER_GUIDE.md` 的经验：Xet 使用保守并发和较长超时；连续解码/重建失败时改用固定 `.aria2` 状态文件的 HTTP Range 续传；不使用文件逻辑大小判断完成，只有固定 revision 的 SHA256 校验通过后才写 `assets.complete`。教师大分片可用镜像作为传输通道，但最终完整性仍由 Hugging Face 固定 revision 的 LFS 哈希约束。

## 下一阶段

- 先增加 hit/miss token loss 与 G checkpoint 的 `memory-off` 推理消融，再扫描教师层、注入层及向量尺度。
- 若低成本诊断显示教师表在命中位置稳定优于随机表，再运行 3 seeds × 10M tokens；两项对照都稳定为正后再扩到 50M。

## 固定源版本

| 输入 | 仓库 | revision |
|---|---|---|
| 学生 | Qwen/Qwen3-0.6B-Base | da87bfb608c14b7cf20ba1ce41287e8de496c0cd |
| 教师 | Qwen/Qwen3-8B-Base | 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 |
| 原料 | openbmb/Ultra-FineWeb | 02c85641e3d19a854be2e09139c25adaa9518063 |
| C-Eval | ceval/ceval-exam | 617524a00b307ff6f9933702f724131fe12ca7ce |
| CMMLU | haonan-li/cmmlu | efcc940752ea4a1ea94d2727f11f83858d64fc8e |

服务器 `assets.lock.json` 保存完整文件清单，`data/processed/manifest.json` 保存数据统计，`environment.freeze.txt` 保存实际 Python 包版本。
