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

## 判断

显式 alignment 改变了结论。未对齐机制在 CMMLU 上 G−S 为 −0.707；对齐后 low-LR G−S 在三个 seed 上一致变为正，且正确映射同时优于 shuffled、LoRA 和随机投影。这四个对照共同表明收益来自教师—学生对应关系，而不只是额外参数或 N-gram ID。

当前证据仍限于三个 seed、一个教师/学生层位和约 +0.23 个百分点。下一步应先将 low-LR aligned 配置扩展到 5M，并保留相同 CMMLU 主指标；同时加入 aligned memory on/off 和不同学科分组分析。只有 5M 仍保持 G−S/G−L 为正，才值得增加表规模或测试其他层位。
