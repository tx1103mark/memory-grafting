# 显式教师—学生 Alignment 实验

本阶段先做离线判别，不直接继续下游训练。固定 Qwen3-8B-Base block 12 教师表与 Qwen3-0.6B-Base block 1 学生目标，对 60k 个确定性抽样 key 做 70/15/15 key-disjoint 划分。PCA/whitening 只使用 train keys，方法和目标只看 validation，最终报告 held-out test。

第一阶段筛选 raw、center、remove-top-16、PCA-256、whiten-256、whiten-256+L2。每种方法分别训练正确表和固定 shuffled 表的同配方线性投影。第二阶段对 validation margin 最好的变换，使用三个 projection seed 比较 cosine 与 cosine+InfoNCE。

指标包括 held-out cosine、normalized MSE、CKA、Recall@1 和 Recall@10。通过标准是三个 projection seed 的正确表 test cosine 均高于 shuffled；下游实验前还要检查检索 margin，而不能只依赖 cosine。

通过后导出 1024 维 `aligned_table.pt`。下一阶段只测试 T12→S1，不启用 fallback/ShortConv，比较 frozen projection 与低学习率 projection，并以 CMMLU 为主指标。

## 离线结果

60k keys 的离线门槛通过。验证集选择了 mean-centering 与 cosine+InfoNCE；PCA-256 和 whitening-256 均未超过简单中心化。三个 projection seed 的 held-out 结果为：

| 映射 | Cosine | Recall@1 | Recall@10 | CKA |
|---|---:|---:|---:|---:|
| Correct | 0.7821–0.7825 | 93.70%–94.04% | **99.56%–99.61%** | 0.899–0.901 |
| Shuffled | 0.3033–0.3045 | 0%–0.05% | 0.49%–0.59% | 0.058 |

这证明教师行与对应学生表示之间存在强、可泛化的映射，而 shuffled 表无法通过相同训练配方恢复对应关系。

## 2M 下游结果

对齐表被投影到学生 1024 维空间。`frozen` 使用冻结 identity K/V；`low-lr` 从 identity 初始化并以 2e-6 学习率更新；`random` 使用同一对齐表但随机初始化 K/V。所有配置关闭 fallback 与 ShortConv。

CMMLU 是预注册主指标。表中为三个 seed 的学科 macro accuracy 均值±标准差。

| 配置 | 0.5M | 2M |
|---|---:|---:|
| Frozen G | 51.084±0.098 | 51.049±0.047 |
| Frozen S | 51.062±0.048 | 50.947±0.214 |
| Low-LR G | **51.120±0.078** | **51.100±0.029** |
| Low-LR S | 51.030±0.099 | 50.862±0.014 |
| Random-projection G | 51.109±0.106 | 50.951±0.118 |
| LoRA | 51.046±0.021 | 50.870±0.019 |

2M 的关键配对差异：

- Low-LR G−S：`+0.238`，三个 seed 为 `+0.242/+0.218/+0.253`，95% t CI `[+0.194,+0.282]`。
- Low-LR G−L：`+0.229`，三个 seed 为 `+0.229/+0.215/+0.244`，95% t CI `[+0.194,+0.265]`。
- Frozen G−L：`+0.179`，三个 seed 为 `+0.210/+0.153/+0.175`，95% t CI `[+0.109,+0.250]`。
- Low-LR G−random projection：`+0.149`，三个 seed 方向均为正。

0.5M 时各组基本无法区分，优势在继续训练到 2M 后形成。C-Eval validation 在 2M 没有提升：low-LR G−S 为 +0.050、G−L 为 −0.225。这与本轮预注册的 CMMLU 主指标不冲突，但说明收益仍然较小且具有任务差异。

## 5M-token 预注册确认

2M-token 结果确定后，固定 T12→S1、500k 表、关闭 fallback/ShortConv，并只保留表现最好的 low-LR aligned 配置。5M 实验从相同 Qwen3-0.6B-Base 权重独立训练，不从 2M checkpoint 续训；比较正确教师表 G、打乱教师表 S、LoRA-only L，使用 seeds 42/43/44。CMMLU 是唯一主指标，C-Eval 是次指标。主要检验为配对的 G−S，其次为 G−L；只有三个 seed 均同号且 95% t 区间不跨 0，才视为复制 2M 信号。

| 配置 | CMMLU 5M | C-Eval 5M |
|---|---:|---:|
| Low-LR G | 50.871±0.082 | 50.158±0.591 |
| Low-LR S | 50.946±0.118 | 50.203±0.208 |
| LoRA | 50.902±0.101 | 50.354±0.509 |

CMMLU 的 G−S 三个 seed 为 `−0.163/+0.022/−0.085`，均值 `−0.075`，95% t CI `[−0.306,+0.156]`；G−L 为 `+0.098/−0.096/−0.096`，均值 `−0.031`，95% t CI `[−0.309,+0.247]`。C-Eval 的 G−S 为 `−0.046`，G−L 为 `−0.196`，区间也都跨 0。5M 因而没有复制 2M 的 +0.238 CMMLU 信号。

memory on/off 诊断仍显示正确表被使用：关闭 G memory 后，三个 seed 的 hit-token loss 分别上升 `0.000267/0.000177/0.000296`，且有效注入范数约为 hidden norm 的 1.49%–1.54%。G 的最终 alpha 约 0.004；S 的 alpha 接近 0，有效注入仅约 0.05%–0.11%。这说明优化器学会了识别并采用正确对应关系，但这种局部 next-token loss 收益没有稳定转化成 CMMLU/C-Eval 准确率。

CMMLU 的 67 个学科中，只有 modern Chinese 和 journalism 等 3 个学科在三个 seed 上 G−S 均为正，而 18 个学科三个 seed 均不为正；较大的单学科差值通常只有一至两个 seed 支持。没有出现足以解释整体收益的稳定学科簇。

## 判断

显式 alignment 确实改善了 memory 的可学习性：离线映射、2M CMMLU 和 5M on/off loss 都能区分 G 与 S。但预注册的 5M 下游确认失败，因此不能把 2M 的约 +0.23 点解释为稳定能力提升。现有证据更准确的表述是：正确教师 memory 能被模型识别并降低很小的 token loss，但多选知识评测收益较小、对训练预算敏感，当前接口尚未形成稳健迁移。

下一步不应继续单纯增加相同 causal-training token。更有判别力的是在 2M 附近保存密集 checkpoint，定位收益形成和消失的时间段，并按 memory hit/frequency 与 CMMLU 学科拆分；若要继续改机制，应直接加入与下游预测相关的 alignment/contrastive 目标或可学习校准，而不是扩大表或扫描更多层位。
