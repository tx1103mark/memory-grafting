# Memory Grafting 中文 Pilot 结果

运行时间：2026-09-14。模型为 Qwen3-0.6B-Base，教师为 Qwen3-8B-Base；评测使用 `lm_eval==0.4.13` 的 C-Eval val，0-shot，共 1,346 题。

## 对照定义

- B0：未继续训练的 Qwen3-0.6B-Base。
- L：仅训练 LoRA。
- R：训练相同 LoRA 和注入 adapter，查表内容为冻结随机向量。
- G：训练相同 LoRA 和注入 adapter，查表内容为 Qwen3-8B-Base 第 12 个 block 输出的冻结向量。
- L/R/G 使用完全相同的数据顺序、seed 42、优化配置和 2,000,000 target-token 预算。最长后缀查找顺序为 4→3→2-gram，训练中匹配率约 23%–30%。

## 结果

| 组别 | C-Eval 加权 accuracy | 学科宏平均 | 最终验证 loss | 最终 alpha | 最终 gate bias |
|---|---:|---:|---:|---:|---:|
| B0 | 50.9658% | 49.7481% | — | — | — |
| L | 50.6686% | 49.5247% | 2.9145763 | — | — |
| R | 50.2972% | 49.0841% | 2.9145052 | 0.0025243 | -1.9986736 |
| G | 50.5201% | 49.2719% | 2.9145021 | 0.0065544 | -1.9978766 |

配对比较：

| 比较 | 学科宏平均差值 | 95% bootstrap CI | 改对 / 改错题数 |
|---|---:|---:|---:|
| G − L | -0.2528 pp | [-1.0361, 0.5303] pp | 15 / 17 |
| G − R | +0.1877 pp | [-0.6413, 1.0039] pp | 16 / 13 |

每个区间使用 seed 42 的 10,000 次配对 bootstrap。单次训练的区间只反映评测样本不确定性，不包含训练 seed 方差。

## 判断

本轮 pilot **没有提供 Memory Grafting 提升 C-Eval 的统计证据**：G 低于 L，且 G−L、G−R 的区间都跨 0。G 的验证 loss 仅比 R 低约 `3.0e-6`，不能视为有意义的差异。

教师表的 `alpha` 增长到随机表的约 2.6 倍，说明优化器对教师向量和随机向量采取了不同权重；但这个内部信号没有转化为 C-Eval 增益。L/R/G 都低于 B0，表明 2M tokens 的通用语料继续训练本身在该评测上略有负迁移，也限制了对 graft 的判断。

## 下一步建议

暂不直接扩大到原计划的三种子 × 50M tokens。先用较低成本的诊断实验定位瓶颈：

1. 在同一批验证文本上分别测有命中位置和无命中位置的 token loss，直接判断 G 是否只在 memory hit 上优于 R/L。
2. 增加 `memory-off` 推理消融：对同一个 G checkpoint 在评测时关闭注入。它能区分 G 的收益来自教师表内容，还是训练期间 adapter 对 LoRA 路径的间接影响。
3. 比较教师层 4/8/12、注入层 1/3/6，并统一投影后的 RMS；当前单一层配对可能存在表征层级和尺度不匹配。
4. 将 60k key 从纯文档频次改为“高频且对 student loss 有贡献”的选择：先计算每个 n-gram 的 baseline excess loss，再优先保留高频高损失 key。
5. 若命中位置 loss 显示 G 稳定优于 R，再运行 3 seeds × 10M tokens；只有 G−L 和 G−R 都稳定为正时再扩到 50M。

## 可复核产物

- 服务器目录：`/mnt/disk0/home/tysearch/ngram-embedding`
- 原始 harness 结果：`runs/{B0,L,R,G}_ceval_val.json`
- 每题样本：`runs/{B0,L,R,G}_ceval_val.*.samples.jsonl`
- 配对结果：`runs/G_vs_L_pilot.json`、`runs/G_vs_R_pilot.json`
- 训练曲线：`runs/{L,R,G}_seed42_2000000tokens/metrics.jsonl`
- 完成标记：`runs/pilot.complete`
