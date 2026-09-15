# Qwen3 Base Memory Grafting

研究 Qwen3-8B-Base 的离线局部表示，能否改善 Qwen3-0.6B-Base。详细假设与评测协议见 [TRAINING_PLAN.md](TRAINING_PLAN.md)，完整消融、结果和结论见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。本仓库是可验证的实验实现，不预设一定有提升。

当前结果显示：初版 60k frozen teacher table 没有稳定教师语义收益；扩展为 500k entries 并加入 trainable fallback 与因果 ShortConv 后，T12→S1 在单 seed C-Eval 上比 LoRA 高 1.36 个百分点，但 CMMLU 未复现。该结果需要多 seed 独立确认。

## 对照

| 组 | 起点 | 训练参数 | 记忆 |
|---|---|---|---|
| B0 | 0.6B Base | 无 | 无 |
| L | 同一 Base | LoRA | 无 |
| R | 同一 Base | LoRA + adapter | 冻结随机向量 |
| G | 同一 Base | LoRA + adapter | 冻结 8B Base 第12块输出 |

所有输入都是普通文本；不使用 chat template、SFT 标签或思维链。原文经过 tokenizer 得到 token IDs；以右移一位的 token 为标签，padding 和重叠上下文对应标签设为 `-100`。

## 安装及资源

在有 CUDA PyTorch 的独立环境中：

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets --root .
python -m scripts.prepare_data --root .
python -m pytest -q tests
```

数据源和模型 revision 自动写入 `assets.lock.json`。只下载一个固定随机中文分片、完整学生 Base、教师 embedding 与 blocks 0–11 所需的前两个权重分片，以及评测文件。一个 Ultra-FineWeb 中文分片足以覆盖本轮5000万训练 target tokens；该选择降低了原料来源多样性，manifest 会明确记录。默认另留100万验证 target tokens；文档不截断，因此原料略超预算，训练器再精确控制预算。构表只统计 train 文档，默认每阶2万条2/3/4-gram，共6万条。选不满时直接报错。

预处理包含中文比例/最短长度筛选、规范化精确去重、近似 MinHash 去重、评测题及选项的规范化30字符重叠排除。它不是全面的语义去污染。使用评测内容的唯一预处理目的是排除训练污染；不要按测试题选择记忆 key 或训练样本。

`scripts/bootstrap.sh` 专供当前 `h200` 服务器，复用已有 CUDA PyTorch 建立独立 venv，不修改原环境。其他机器应按上述通用命令安装。

## 构表和训练

先设置已分配的 GPU；本仓库不自动抢占或终止其他进程：

```bash
export CUDA_VISIBLE_DEVICES=YOUR_ALLOCATED_GPU
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
python -m scripts.build_memory --root .
python -m scripts.train --root . --group L --seed 42 --tokens 2000000
python -m scripts.train --root . --group R --seed 42 --tokens 2000000
python -m scripts.train --root . --group G --seed 42 --tokens 2000000
```

当前服务器也可执行 `bash scripts/run_pilot.sh`，依次完成测试、构表、B0评测、L/R/G各200万token训练及C-Eval val评测。正式多种子实验应在 pilot 的有效性和实现检查完成后再运行：

```bash
for seed in 42 43 44; do
  for group in L R G; do
    python -m scripts.train --root . --group "$group" --seed "$seed" --tokens 50000000
  done
done
```

所有训练运行从同一 Base 重置，不接着 pilot 继续训练。单卡 micro-batch=2、梯度累积=8、序列最长1024。梯度按整个累积批次有效 target 数加权。LoRA与门控适配器均保留FP32参数；主干/冻结表使用BF16，非重入 checkpoint 保证冻结后缀仍传播梯度。

```bash
python -m scripts.train --root . --group G --seed 42 --tokens 2000000 \
  --resume runs/G_seed42_2000000tokens/last.pt
```

恢复要求数据、记忆及训练配置一致。checkpoint 保存可训练权重、optimizer、随机状态和数据游标；依赖锁定的 Base 权重及 `memory/table.pt`，不是独立完整模型。

## 评测

```bash
# Pilot：官方 ceval-valid 任务读取 C-Eval val；不用 CMMLU test 反复调参。
python -m scripts.evaluate --root . --tasks ceval-valid --output runs/B0_ceval_val.json
python -m scripts.evaluate --root . --checkpoint runs/G_seed42_2000000tokens/last.pt \
  --tasks ceval-valid --output runs/G_ceval_val.json
# 正式冻结方案后：显式 C-Eval test 变体及官方 CMMLU test
python -m scripts.evaluate --root . --checkpoint runs/G_seed42_50000000tokens/last.pt \
  --tasks ceval-test,cmmlu --output runs/G_seed42_final.json
```

评测统一使用 `lm_eval==0.4.13`，继承官方 HFLM 的分词、候选评分与批处理；仅替换模型前向。官方 C-Eval/CMMLU 的学科说明、模板、delimiter、acc/acc_norm 和聚合配置保持原样，`--num-fewshot 0` 为本轮固定设置，不使用 chat template。数据读取改为已锁定版本的本地 JSON，任务名加 `graft_` 前缀以区分数据快照；原始任务配置哈希随报告保存。

当前记忆适配器固定 `--batch-size 1`。harness 的 causal `_model_call` 不传每条样本原始长度，批量左侧 padding 可能形成伪 n-gram；脚本会拒绝更大 batch，避免悄悄改变查表结果。所有组采用相同 batch size。

输出 harness 完整 results、逐题 samples、官方按题量加权结果以及额外的学科宏平均。`--limit N` 是每个学科最多 N 题，仅供冒烟测试，报告明确标为 partial。`ceval-valid` 使用 val；若它用于 pilot 调参，只能视为开发集结果。`ceval-test` 是本项目显式变体：复用官方模板/指标，仅将 split 改为已核实带标签的 C-Eval test，不冒称标准 ceval-valid。不同版本/prompt/few-shot 的排行榜不能直接比较。

G checkpoint 支持 `--memory-mode zero` 检查同一模型关掉记忆后的变化。实现仅支持完整前向评分，不支持带记忆的 KV-cache 生成；不要据此声称线上生成加速。

## 产物与日志

- `assets.lock.json`：固定源仓库 revision 与分片。
- `data/processed/manifest.json`：实际数据量、过滤统计、评测数量。
- `data/processed/keys.json`：有序 key、短语及文档频次。
- `memory/table.pt`：teacher/random表；第0行严格为零。
- `runs/<group>_seed<seed>_<budget>tokens/metrics.jsonl`：loss、梯度范数、命中率、alpha、验证loss、速度。
- `last.pt`：固定训练预算末状态；`best.pt`：独立文本验证loss最低状态。
- `complete.json`：该训练预算已完成；没有这个文件不能声称训练完成。

CPU单元测试使用随机初始化的微型Qwen3，只验证实现正确性；不代表实际Base模型性能。实际运行状态另见 `RUN_STATUS.md`。
