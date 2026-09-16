# T12→S1 三种子确认与机制归因结果

更新时间：2026-09-16。15 个新训练任务、15 个 CMMLU、15 个 C-Eval validation 和 13 个非 LoRA memory 诊断全部完成，`errors.json=[]`。本轮固定上一轮机制修复配置，不再搜索层位；CMMLU 是预注册主指标，C-Eval validation 是次指标。seed 42 复用上一轮 checkpoint，seeds 43/44 独立重训。

## 三种子主比较

分数为学科 macro accuracy 百分比，`±` 后为三个 seed 的样本标准差。

| 任务 | full G | full S | L | G−S | G−L |
|---|---:|---:|---:|---:|---:|
| **CMMLU（主）** | 50.122±0.682 | **50.829±0.020** | **50.902±0.101** | −0.707 | −0.780 |
| C-Eval validation | **51.100±1.472** | 50.862±0.268 | 50.354±0.509 | +0.238 | +0.747 |

CMMLU 每 seed 的 G/S/L 为：

| Seed | G | S | L | G−S | G−L |
|---:|---:|---:|---:|---:|---:|
| 42 | 50.174 | 50.851 | 50.820 | −0.677 | −0.646 |
| 43 | 50.777 | 50.814 | 51.014 | −0.036 | −0.237 |
| 44 | 49.416 | 50.823 | 50.872 | −1.407 | −1.456 |

C-Eval 每 seed 的 G/S/L 为：

| Seed | G | S | L | G−S | G−L |
|---:|---:|---:|---:|---:|---:|
| 42 | 52.157 | 51.131 | 50.796 | +1.026 | +1.361 |
| 43 | 51.725 | 50.860 | 50.468 | +0.864 | +1.257 |
| 44 | 49.419 | 50.596 | 49.797 | −1.176 | −0.378 |

三个样本的 t 区间很宽：CMMLU G−S 95% CI 为 [−2.411, +0.997]，C-Eval G−S 为 [−2.812, +3.288]。比区间更有判别力的是方向：主指标 CMMLU 三个 seed 的 G−S 全为负；C-Eval 的正向信号在 seed 44 反转。因此上一轮单 seed 的 +1.03 点 G−S 不能复现为稳定效果。

## 组件消融

下表为三个 seed 的均值；差值使用 `full G − ablation`，正值表示完整机制更好。

| 配置 | CMMLU | C-Eval | full−ablation CMMLU | full−ablation C-Eval |
|---|---:|---:|---:|---:|
| full G | 50.122 | 51.100 | — | — |
| no fallback | 50.260 | 50.913 | **−0.138** | +0.187 |
| no ShortConv | 50.300 | 51.128 | **−0.177** | −0.028 |
| fallback only（关闭教师表） | **50.866** | 49.892 | −0.744 | +1.209 |

CMMLU 上，full G 相对 no-fallback 的配对差值为 `−0.138`，95% CI [−0.232, −0.044]；相对 no-ShortConv 为 `−0.177`，[−0.244, −0.110]。两个组件在三个 seed 上都造成小幅下降。fallback-only 与 S/L 接近且方差很小，而加入正确教师表后均值更低、方差显著增大。

C-Eval 上关闭 fallback 或 ShortConv 基本不改变均值；正确教师表相对 fallback-only 平均高 1.209 点，但区间 [−1.932, +4.349]，且 seed 44 不成立。这更像开发集和训练 seed 共同造成的不稳定任务信号。

## 结论与决策

机制修复没有形成跨 seed、跨评测集的教师语义收益。500k 表、fallback 和 ShortConv 能让支路参与计算，但普通 causal continued pretraining 没有可靠地把可对齐的教师 latent 转成考试能力；CMMLU 反而更支持 shuffled/fallback-only/LoRA。

按预注册规则，不再扩大相同目标下的训练 token 数，也不继续扫描层位。下一阶段应先做显式 teacher-student alignment：

1. 对 teacher table 做中心化与 whitening，消除各向异性和高频公共方向。
2. 用 held-out n-gram keys 训练投影，使教师行预测对应学生 hidden；正确/打乱表使用完全相同配方。
3. 比较 cosine/MSE、CKA、next-token KL 与检索区分度，选择在 held-out keys 上稳定优于 shuffled 的目标。
4. 仅将通过离线判别的投影接回 0.6B，先跑 0.5M/2M 的 G/S/L 小实验；CMMLU 继续作为主指标。

完整机器可读统计见 `remote_results/confirmation_study/summary.json`，单次评测和诊断 JSON 位于同一目录。
