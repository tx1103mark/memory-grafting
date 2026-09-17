# 中文 Memory Grafting 继续训练实验方案

版本：v0.3-base-harness，2026-09-14。代码实现与数据下载已启动，实际完成状态见 [RUN_STATUS.md](RUN_STATUS.md)。以下研究设计仍区分计划与已验收产物，可运行入口见 [README.md](../README.md)。

**v0.3 评测变更（优先于下文 v0.2 的手写评分接口）**：按用户要求统一使用 `lm_eval==0.4.13`。继承 HFLM 的分词、候选评分与聚合逻辑，只替换模型前向以接入记忆。任务来自该版本的 `ceval-valid`（C-Eval val）和 `cmmlu`（CMMLU test）；保留官方学科说明、模板及 delimiter，设 `num_fewshot=0`、关闭 chat template。固定下载版本转成本地 JSON，只修改任务的数据读取路径与 `graft_` 名称前缀，并保存原始任务文件哈希。官方 acc/acc_norm 和按题量加权聚合全部保留，另列学科宏平均。

本轮不把标准 `ceval-valid` 冒称 C-Eval test。已核实固定 C-Eval 快照含 12,342 道带标签 test 题，因此另建显式 `graft_ceval-test` 任务：复用 harness 官方 C-Eval 模板/指标，仅将评测 split 从 val 改为 test，并记录这一差异。Pilot 使用 ceval-valid；配置冻结后使用 ceval-test 和 CMMLU test。C-Eval val 若参与选择，成绩仅作为开发集结果。

目标：以 Qwen/Qwen3-0.6B-Base 为学生，通过中文无标签文本继续训练，验证 Qwen/Qwen3-8B-Base 离线生成的冻结 n-gram 记忆是否带来额外收益。主评测为 C-Eval 与 CMMLU。

本版按用户确认改为 Base → Base，优先研究 Memory Grafting 本身的有效性。所有对照从同一公开 Base checkpoint 开始；评测使用普通文本补全，不使用聊天模板。此前 v0.1 的 SFT 起点和聊天评测协议已被本版替代，二者结果不可直接混合。Base 起点仍属于继续预训练，并非从随机初始化开始训练。

本方案中的学习率、层位、条目数、训练规模是待验证的起始配置，不是论文最优参数。Memory Grafting 原论文主要验证从头预训练，本实验研究已训练模型的后续适配，不能直接承诺复现论文增益。[1]

## 1. 实验问题与范围

### 1.1 需要回答的三个问题

1. 相同中文数据和训练预算下，教师记忆是否优于普通 LoRA 继续训练？
2. 相同记忆 key、维度、连接结构下，教师向量是否优于固定随机向量？
3. 提升是否覆盖多个学科，且存储、推理成本可以接受？

主假设：教师对局部短语的预训练表示，经过学生侧适配后，能提供超出新增适配器和固定短语标识的有效信息。

### 1.2 第一版范围

- 只用中文正文；不添加选择题 SFT、推理链蒸馏或新业务 QA。
- 记忆 key 是学生 tokenizer 的原始 2/3/4-gram token 元组。
- value 是教师离线编码短语后，第 12 个 Transformer block 输出的最后一个有效 token 表示。
- 一张冻结表，在学生第 3 个 block 后注入一次。
- 精确最长后缀匹配；未命中输出零。
- 不使用 Engram fallback、短卷积、在线教师、巨型源模型记忆表或多层注入。
- 第一版用 Hugging Face/PyTorch 实现；部署优化后置。

这些简化是为隔离教师记忆的增量效果，不声称完整复现论文结构。

### 1.3 不应据此得出的结论

- C-Eval、CMMLU 只能反映中文学科知识和部分推理能力，不代表全部中文能力。
- loss 下降、CKA 升高、门控非零，均不能单独证明知识迁移成功。
- 预训练教师和学生的历史数据不可完全审计，本项目只控制新增数据污染。
- 失败只能说明当前预算和配置下未获得收益，不足以否定整个方法。

## 2. 运行前必须确定的输入

| 项目 | 要求 | 当前状态 |
|---|---|---|
| 学生 checkpoint | Qwen/Qwen3-0.6B-Base，固定实际 commit SHA、完整权重与对应 tokenizer | 模型名确定，revision/缓存路径待锁定 |
| 学生权重形式 | 原始 Base 完整权重；不加载已有 SFT 或其他后训练 adapter | 加载时核实 |
| 教师 | Qwen/Qwen3-8B-Base，固定实际 commit SHA | 模型名确定，revision 待锁定 |
| 中文数据 | openbmb/Ultra-FineWeb，锁定中文 split/文件及正文列 | 获取元数据时核实 |
| 评测数据 | C-Eval dev/val/test；CMMLU dev/test 的实际版本和题目数 | 实现前锁定 |
| 运行资源 | 指定空闲 GPU、独立环境、持久数据盘 | 启动前检查 |
| 通用中文回归 | 使用独立 text_validation 跟踪语言建模能力 | 与训练语料同时准备 |

必须核实模型 ID 包含 `-Base`，不能自动回退到不带该后缀的后训练版本。学生和教师的 tokenizer 均来自各自锁定的 Base 仓库，不假设同系列就可交换 tokenizer。现有业务 SFT 模型移植留到后续独立实验，本轮不再依赖用户 SFT checkpoint 路径。

服务器说明见 [H200_SERVER_GUIDE.md](../H200_SERVER_GUIDE.md)。该指南中的磁盘和 GPU 状态是历史快照，实际启动前重新检查；本方案不固定占用某张卡，也不自动连接服务器。

## 3. 数据准备

### 3.1 原料与统一格式

第一版从 Ultra-FineWeb 中文部分做确定性抽样。该数据有中英文文本，实际正文列可能为 `content`，不能假设原始数据一定叫 `text`。[2]

先检查所锁定版本的 split、文件列表、字段、来源信息和许可，再转换为项目格式：

```json
{"doc_id":"sha256:...","text":"中文正文……","source":"...","source_file":"...","source_row":123,"quality_score":0.9,"split":"train"}
```

- `text` 为原始可读正文；规范化去重文本单独计算，避免污染原始标点、公式和大小写。
- `doc_id` 根据规范化正文生成；若来源提供稳定 ID，同时保留。
- 来源缺失时明确记为 unknown，不编造 URL 或学科标签。
- 无需人工标签；`labels` 在因果语言建模时从 token 序列生成。

第一轮不依赖学科分类器。随机抽查约 200 篇文档，记录明显噪声、语言混杂、乱码和模板比例；质量分值可用于分析，但第一版不假设其跨来源可直接比较。

### 3.2 去重与基准污染控制

顺序：原文提取 → 基本清洗 → 精确/近重复聚类 → 基准污染过滤 → 按文档簇划分 → token 预算抽样。

清洗要求：

- 去除空文档、明显乱码、纯导航/广告、机械重复正文；规则与阈值写入 manifest。
- 保留中文学科中的英文术语、数值、LaTeX 和代码片段，不按“非中文字符”一刀切。
- 完全重复只留一份。近重复可先用字符 shingles + MinHash 检索候选，再核实相似度。
- 若一组文档被判定近重复，整簇必须落到同一 split；禁止同文不同片段跨训练和验证。

基准去污染要求：

- 使用锁定版 C-Eval 与 CMMLU 的题干和选项建立排除索引，包括计划作示例的 dev。
- 对题干、选项组合、去掉选项字母后的文本做精确及近重复候选检索。
- 过滤原题、近重复题和包含明显原题解析的文档。初始候选规则可用连续 30 个规范化字符匹配或字符 5-shingle 高相似；它们只是候选规则，需抽查误杀，不能视作污染证明。
- 不因“牛顿第二定律”这类常见概念重叠而删除全部相关知识。
- 对匹配命中的文档簇整体处理，记录过滤原因、题目 ID、相似度和处理结果。
- 测试文本只能用于排除污染，不用于补充 key、选学科、生成变体或生成教师向量。
- 该过滤不能保证发现全部语义改写，更不能排除教师/学生历史预训练污染；在报告里说明。

构表与继续训练必须使用同一份过滤后的训练语料；不能训练数据已去污染，而构表仍用未过滤原文。

### 3.3 划分与预算

先按文档簇哈希分配约 98% train、2% text_validation，随后在各 split 内确定性抽样。若验证文本不足，继续采样候选文档，而不是从训练文档截出验证片段。

| 数据产物 | 目标规模 | 用途 |
|---|---:|---|
| train_full | 约 50M 学生正文 tokens | 正式构表和继续训练 |
| train_pilot | train_full 中确定性的约 2M-token 子集 | 冒烟后的小预算训练 |
| text_validation | 约 1M 学生正文 tokens | loss/PPL、查表覆盖和 checkpoint 选择 |
| C-Eval val | 固定版完整 val | pilot 配置检查，不训练 |
| C-Eval test | 固定版完整 test | 最终评测 |
| CMMLU test | 固定版完整 test | 最终评测 |

预算按学生 tokenizer 的实际 token 数统计，不按字符、文档数量或去重词表大小估算。不下载全量语料；打乱 shard 顺序、跨多个 shard 抽样，保存顺序与随机种子，避免只取数据流开头。

在达到目标预算时保留完整文档，允许略微超额并报告实际值；训练的有效 target-token 数另外统计。不能用 padding 数量填满“50M”。

### 3.4 无标签训练样本与边界

第一版推荐“单文档内分块 + 动态 padding”，不跨文档 packing，序列长度上限 1024。

- 每篇文档显式追加一个固定的结束 token，记录实际 token ID；排除结束 token 参与 n-gram key。
- 长文档切块时最多保留 3 个前缀重叠 tokens，使 4-gram 在块边缘仍有上下文。
- 重叠前缀只提供上下文，其 label 设为 -100，防止重复训练同一目标。
- padding 的 label 设为 -100；真实 EOS 即使与 padding ID 相同，也不能按 token ID 全部屏蔽，应按 attention mask 区分。
- 保持位置定义一致：位置 t 的记忆来自截止 x_t 的后缀，位置 t 的 logits 预测 x_(t+1)。
- 使用标准 Qwen CausalLM loss 时 labels 与 input_ids 同序，内部完成 shift；自写 loss 才显式使用 logits[:, :-1] 和 labels[:, 1:]。禁止重复 shift。
- 统计有效监督数时统计 shift 后的非 -100 labels；区分正文 tokens、输入 tokens、padding tokens 和监督 targets。
- 全部实验使用相同文档顺序、分块和 target mask。

如后续改用 packing，须实现 segment 边界、禁止跨文档的记忆后缀，并决定是否使用 block-diagonal attention。packing 是新版本配置，不与第一版结果混用。

### 3.5 数据验收产物

- `corpus_manifest.json`：原始版本、文件、字段映射、抽样种子、清洗规则、SHA256、各 split token 数。
- `decontamination_report.json`：排除规模、规则、人工抽查结果。
- `train.jsonl`、`text_validation.jsonl` 或等价 Arrow/Parquet。
- `tokenized/`：input_ids、attention_mask、labels、doc_id、块起始位置、有效监督数。
- 文档簇无跨 split 泄漏检查；数据流可复现检查。

## 4. 离线记忆表

### 4.1 key 选择

用 train_full 的学生原始 token 序列统计，不用验证集或测试集。不跨文档、不跨 EOS。对每个 key 记录 count 与 document frequency；同一文档重复出现只增加一次 document frequency。

第一版总计 60,000 条：2-gram、3-gram、4-gram 各 20,000。按 document frequency 降序，count 次之，token 元组字典序作为确定性的 tie-break。

- 候选过滤纯特殊 token、纯空白/标点、无法可靠恢复文本的字节片段。
- key 保持原 token 元组，不能通过教师 tokenizer 重编码后替换。
- 表面文本保留前导空格和标点，不使用可能改写片段的自动 cleanup。
- 如中文 token 元组解码后产生替换字符，第一版丢弃并记录数量；即便不同 key 对应相同文本，也保留独立映射或显式共享 value，不能无记录合并 key。
- 某阶有效候选不足 20,000 时，使用全部有效候选，报告真实条目数，不用噪声凑数。

### 4.2 教师向量生成

教师：固定 revision 的 Qwen/Qwen3-8B-Base，eval 模式、BF16、无梯度，不使用聊天模板或生成过程，只编码短语文本。若此前曾生成后训练教师的向量，不能复用为本版 Base 记忆，需重新构表并更新 manifest。

层位统一使用以下规范：

- `teacher_block_ordinal: 12` 表示第 12 个 block，模块列表 index 为 11。
- 提取该 block 的 residual 输出，不是 final RMSNorm 后的输出。
- 若使用 output_hidden_states，必须在锁定 Transformers 版本上验证元组位置含义，不能只凭数字猜测。
- 使用 attention_mask 定位最后一个有效 token。建议右 padding，显式固定特殊 token 添加规则；若 `add_special_tokens=False`，记录为 manifest 属性。
- 记录向量 shape、dtype、范数分位数、NaN/Inf 数量、模型/代码 revision。

构表时 memory value 是教师对短片段的表示，不包含该片段在原文中前后段落的所有知识。若后来加入真实上下文聚合，需另设实验，不能静默替换。

可在教师的第 12 层截断 forward 以节省计算，但先在少量短语上验证截断结果与完整 forward 对应层一致。不需要保存其他层、全词表 logits 或生成答案。

### 4.3 存储及访问

预计 60,000 × 4,096 × 2 = 491,520,000 bytes，约 492 MB / 469 MiB，不含索引。维度应从实际配置读取，不硬编码。

产物：

```text
memory/
  keys.parquet              # key、频次、doc_freq、表面文本、row_id
  teacher_values.safetensors
  random_values.safetensors
  memory_manifest.json
```

第一版将冻结表放在 GPU 上，避免引入 CPU offload 的带宽干扰。以后如量化或压缩，需单独评估精度与延迟。

row 0 可留作 miss 零向量，真实条目从 row 1 开始；命中掩码必须显式存在。

精确查找采用 4 → 3 → 2 的最长后缀优先级。哈希只作索引加速，必须验证完整 key 或使用无歧义的键结构，不能把哈希碰撞当命中。

训练集可离线预计算每个位置的 memory_id；验证/测试使用同一查找实现在线计算或按完整提示预计算，不扩充表。被 padding 或跨边界的位置必须为 miss。

### 4.4 固定随机记忆对照

R 与 G 使用完全相同的 key、维度、miss 行、适配器和查表规则。随机表可根据教师表的逐维均值/标准差抽样高斯向量，再按相同 key 的教师行范数缩放，以减少尺度差异；该方法保留部分统计信息，须在 manifest 中披露。随机表生成使用独立固定种子，训练期间冻结。

不能让随机表训练、教师表冻结，也不能随机表维度更小。适配器初始化随机流与表生成随机流分开，避免初始化被生成表消耗的 RNG 改变。

## 5. 学生结构和梯度路径

### 5.1 最小注入模块

在学生第 3 个 block 后（模块 index 2），下一 block 的输入前执行：

```text
m_t = frozen_table[memory_id_t]
e_t = RMSNorm(m_t)
k_t = W_k e_t
v_t = W_v e_t
g_t = sigmoid(dot(RMSNorm(h_t), RMSNorm(k_t)) / sqrt(d_student) + b)
delta_t = alpha * hit_t * g_t * v_t
h'_t = h_t + delta_t
```

第一版使用无额外可学习增益的 RMSNorm，epsilon 与实际数值类型兼容；W_k/W_v 无 bias，gate bias 初始 -2，alpha 为可学习标量、初始 1e-3。W_k/W_v 采用固定种子的普通非零初始化。

最终 `delta` 强制乘 hit mask，因此 miss 严格零。不要同时将 alpha 与所有投影权重归零。上述门控是工程起点，需要在 pilot 中检查是否饱和、delta 是否异常。

教师维度 4096、学生维度 1024 时，两投影约 8.39M 参数，另外仅少量标量。这些参数必须计入总可训练容量和推理计算。普通 LoRA 基线与记忆组并非严格等参数；R/G 是主要等结构对照，可在后续加入约等参数 MLP adapter 基线。

### 5.2 训练参数管理

- 主训练：学生原始权重冻结，LoRA 和记忆投影/门控可训练，表冻结。
- LoRA 初始配置：r=16、alpha=32、dropout=0，目标为 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj；必须核实实际模块名和数量。
- 学生无记忆组使用同样 LoRA 设置。
- 保留学生原始 lm_head 与 embedding 的参数共享关系，不额外解冻它们。
- 新增模块注册为 nn.Module，随模型设备、state_dict 和 train/eval 状态管理，不作为孤立 Python 对象保存。

冻结权重不等于不需要梯度路径：注入之后的学生层需要对输入求导。不能对整个学生 forward 用 no_grad。教师构表可以 no_grad，冻结表读取也不需要表参数梯度。

如开启 gradient checkpointing，优先 non-reentrant 配置，验证冻结输入情况下新增模块和 LoRA 仍有梯度。每次使用不同库版本都执行梯度验收。

### 5.3 模型接口契约

建议模型 forward 显式接收 `memory_ids`/`memory_hit`，由自定义 Qwen3 forward 或包装的 decoder layer 传递；正式实现避免依赖全局可变 hook cache，否则 checkpoint 重算、并发和多卡可能串数据。

- Trainer 不得移除新增字段；forward signature、collator 和 remove_unused_columns 需联动检查。
- collator 按相同位置 padding memory_ids，不把命中字段误传给原生 Qwen forward。
- 调用输出仍兼容 loss/logits；统计量通过单独接口输出，避免 Trainer 将其误当预测 logits。
- 自定义 checkpoint 保存基座标识、LoRA、新模块、表 manifest/key/value 的引用与校验和。
- 推理必须恢复完整组合，仅 merge LoRA 不足以恢复记忆模型。

## 6. 实验矩阵与训练预算

### 6.1 主实验

| ID | 起点 | 更新内容 | 记忆内容 | 用途 |
|---|---|---|---|---|
| B0 | 固定 Qwen3-0.6B-Base checkpoint | 不训练 | 无 | 原始 Base 能力参考 |
| L | 同一 Base checkpoint | LoRA | 无 | 中文继续训练收益 |
| R | 同一 Base checkpoint | LoRA + 记忆适配器 | 固定随机表 | 新容量/固定 key 特征控制 |
| G | 同一 Base checkpoint | LoRA + 记忆适配器 | 固定 Base 教师表 | Memory Grafting 主实验 |

L/R/G 读取完全相同样本、mask 和有效监督 token 预算。R/G 新模块从同一初始化开始；相同 seed 下 LoRA 初始化一致。

R/G 可在 pilot 中先做“仅记忆适配器训练”的诊断，但主实验仍从共同 Base checkpoint 重新初始化，不继承诊断权重。避免 G 多训练一个 warm-up 阶段而 L/R 没有。

可选第二轮：学生自编码表、容量匹配 MLP adapter、训练时固定打乱教师 key–value 对照。不同维度的学生表比较要报告结构差异，不能宣称严格等参数。

### 6.2 默认训练配置

| 参数 | 起始值 |
|---|---|
| precision | BF16，优化器状态按实现默认稳定精度 |
| max sequence length | 1024 |
| 主训练 GPU | 1 张空闲 H200；多卡作为吞吐扩展 |
| 每卡 batch | 起始 2，梯度累积 8，即单卡 16 条/更新 |
| data order | 固定 seed，按文档/块 manifest 复现 |
| LoRA learning rate | 2e-5 |
| memory adapter learning rate | 1e-4 |
| optimizer | AdamW，betas=(0.9,0.95)，eps=1e-8 |
| weight decay | 0.01；bias/norm/标量门控不衰减 |
| LR schedule | cosine；前 5% 更新 warm-up |
| gradient clipping | global norm 1.0 |
| pilot budget | 约 2M 有效 target tokens |
| main budget | 约 50M 有效 target tokens |
| seeds | pilot=42；正式=42/43/44 |
| 训练日志 | 每 10 次 optimizer update |
| 验证与保存 | pilot 中点/末尾；正式每约 5M targets |

单卡 16×1024 是 16,384 输入槽位/更新，不等于有效监督量。约 50M / 16,384 ≈ 3,052 次更新只是无 padding 的粗估；实际更新数由有效 targets、重叠和最后 batch 决定。

预处理后生成共同的 batch manifest 和固定更新数，使 L/R/G 消耗相同 targets。目标预算允许最后一个 batch 略超，记录实际值。若一遍 train_full 不足 50M targets，按固定顺序循环并记录重复量，各组一致。

变长 batch/梯度累积应按有效 target 总数加权损失，不简单平均每个 microbatch 的均值。多卡时按全局有效 targets 正确归一化；用单卡/多卡小样本梯度一致性检查验证。

如果显存不足，所有组统一降低 batch 并相应增加累积；不能只有某组悄悄减少序列长度或训练预算。记录 trainable params、峰值显存、有效 targets/s、GPU-hours。

### 6.3 checkpoint 选择

- 主表报告固定 50M target-budget 的最终 checkpoint，避免各组选不同训练长度。
- 另报 text_validation loss 最低的 checkpoint，所有组使用相同选择规则。
- C-Eval val 仅用于 pilot 中有限的配置比较和明显协议错误诊断，不每几个 step 调参。
- 正式 test 在配置冻结后统一评测。看完 test 后的任何改动均标为新一轮探索，不再把同一 test 当完全未见。
- 不设“必须提升若干分才发布”的筛选规则；失败和退化结果同样记录。

## 7. 评测协议

### 7.1 数据与版本

官方来源：[C-Eval][5]、[CMMLU][6]。必须记录下载 revision、split、题目数、学科映射和数据哈希。C-Eval 官方已宣布公开完整 test，但要核实所用发行版标签和 schema；不能用 val 的分数冒充 test。

本方案默认内部对照协议，不直接声称与榜单设置相同。若使用 lm-evaluation-harness 或 OpenCompass，先检查锁定版本的任务配置是否实际读取 test（部分历史任务使用 validation），以及评分方式和汇总方式。未知 task ID 不写成已经可运行的命令。

### 7.2 主协议：固定 0-shot 选项字符串评分

使用相同普通中文补全提示，末尾 `答案：` 后紧接候选字母，不追加换行或空格：

```text
以下是关于{subject_name}的单项选择题及其正确答案。

{question}
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
答案：
```

直接以学生 Base tokenizer 编码上述普通文本，固定 `add_special_tokens=False`；不调用 `apply_chat_template`，不传入 `enable_thinking`，不插入 system/user/assistant、`<think>` 或其他聊天控制标记。Base 协议不存在“关闭思考”的模型开关。对 B0 在开发侧做少量模板和候选 token 边界检查，保存完整提示字符串与 token IDs 后，所有组固定。即使模型仓库附带聊天模板，也不在本实验使用。

定义候选 continuation 为字符串 `A`、`B`、`C`、`D`，分数：

```text
score(c) = sum_j log p(candidate_token_j | rendered_prompt, candidate_tokens_<j)
prediction = argmax_c score(c)
```

- 不进行自由生成，不计算思维链；候选若多 token，计算完整 continuation 概率。
- 验证 tokenizer 对 prompt/continuation 的边界一致性，不能假设 A/B/C/D 永远单 token，也不能直接取词表里猜测的 ID。
- 不加长度归一化作为主指标；记录每个候选 token 数。如另测完整选项文本概率，单列为不同协议。
- 对评分位置严格因果：预测候选首 token 时，记忆只能依赖 prompt；不能先把候选首 token 加入其自身预测位置的记忆。
- 第一版可禁用 KV cache，对候选做完整 forward，减少 cache 接线错误。C-Eval/CMMLU 的 prompt 查表规则与训练相同。
- 若完整 prompt 太长，优先在支持长度内完整评分；第一版超过评测上限 4096 的题目报错统计并解决，不静默删除选项、跳题或只保留较短题。
- exact tie 按固定规则（A→D）处理并报告 tie 数；所有组 eval 模式，无 dropout。

记录的是“四候选评分准确率”，不混称自由生成准确率。后续如增加 few-shot 或思考模式，另立结果列。

### 7.3 指标

每个基准分别报告：

1. Micro accuracy = 正确题数 / 总题数。
2. Subject-macro accuracy = 各学科 accuracy 的无权平均，固定为内部主指标。
3. 学科逐项成绩及官方大类汇总（按明确权重计算）。
4. 三种 seed 的均值、标准差；G-L 和 G-R 的分数差，以百分点表示。
5. 纠错率：基线错误题中 G 改正确的比例；退化率：基线正确题中 G 改错误的比例，同时给数量。
6. 独立中文 text_validation loss/PPL，检查考试收益是否伴随语言建模退化；原业务 SFT 回归属于后续迁移实验。

官方默认 aggregate 如与内部 macro 不同，两者分别标明。不要只取两个基准均值掩盖其中一项退化。

置信区间：对相同题目 ID 做 paired bootstrap，按学科分层，固定抽样 seed 与重复次数（例如 10,000）。有来源题组时优先组内聚类抽样。单 seed 的题目 bootstrap 不能代替训练 seed 方差。

### 7.4 记忆诊断与干预

- 在 text_validation 和评测 prompt 上统计 token 位置命中率，按最长命中阶数拆分，计数不重复。
- 将题干/选项与 system/模板分开统计，避免“请选择正确答案”带来的高覆盖误导。
- 以固定题目 ID 比较所有组的表现；不把生成长度不同的命中率直接比较。
- 抽样记录 gate、delta norm / h norm、NaN/Inf、不同来源的激活量。
- 训练后对 G 单独做 gate 关闭与 value 替换干预；标记为分布扰动分析，不代替训练时对照。
- CKA 仅作为可选表征分析，不作为成功门槛；不保存所有层全量 logits。
- 选项重排鲁棒性作为补充：固定置换种子，正确重映射答案，所有组同样处理，单列成绩。

### 7.5 延迟与存储

准确率主评测可 full forward/use_cache=False；部署延迟必须使用正确实现的 cache，未完成 cache 接入时只报告 full-forward 性能，不冒称逐 token 推理速度。

固定硬件与 backend，分别测：

- Prefill：batch=1，prompt 长度 128/512/1024；覆盖率按实际样本记录。
- Decode：相同 prompt、固定 128 个新增 tokens，记录停止规则；gate-off 或外部输入 replay 仅作同形状补充。
- 预热后同步 CUDA，报告 p50/p95、显存峰值、表存储、加载时间。
- 明确是否包含分词、查找、CPU→GPU 数据传输；端到端延迟与 GPU kernel 延迟分开。
- CPU offload/量化若未测，结论不能外推。

单层精确 n-gram decode 只需最近 3 个历史 tokens；prefill→decode、batch 请求隔离、beam reorder 和缓存重置均须测试。第一版不支持的生成模式应显式报错。

## 8. 实施步骤与验收

### P0：环境与输入锁定

产物：`environment.lock.json`、学生/教师 manifest、评测 manifest。

验收：学生与 tokenizer 可加载；学生/教师 Base 起点和 revisions 确认；选定库版本能完成 forward/backward；实际 GPU/磁盘可用。固定 Python、torch、transformers、datasets、peft、accelerate、safetensors 和评测代码 commit，不直接沿用滚动 main 版本。

### P1：准备 50M 语料与固定切分

产物：第 3 节数据文件和数据报告。

验收：文档簇隔离、基准污染排除日志、正文列映射正确、token 预算与边界测试通过。构表数据仅来自 train。

### P2：统计 key 与离线构表

产物：60k key、教师表、随机表、两个表的 manifest。

验收：表维度一致、无 NaN/Inf、最后有效 token 提取正确、相同短语批处理/单条处理结果数值一致；冻结属性明确。

### P3：实现训练/评测共用模型

产物：自定义模型、查找器、collator、训练和选择题评分入口。

必须通过的正确性检查：

1. alpha=0 或全部 miss 时，改造模型与同权重原生模型 logits 在合理 BF16 容差内一致。
2. 4/3/2 最长匹配、EOS 重置、跨文档阻断、padding、块开头上下文正确。
3. 修改未来 token 不改变此前位置的 memory_ids 或 logits；只比较此前位置，不把该 token 自身纳入前缀。
4. 冻结表参数没有梯度、不进入 optimizer；投影/门控与预期 LoRA 参数有有效梯度。
5. checkpointing 开关不改变梯度可达性；保存/重载后 logits 一致。
6. loss 的 shift 与 mask 正确，变长 batch 与累积梯度归一化正确。
7. 从已保存组合 checkpoint 重新评测，确实加载记忆，不能退回裸 Qwen。
8. 多 token 候选评分与手算小例子一致，候选首 token 无未来泄漏。

测试的数值容差按精度设定并记录；固定用一批 FP32 小模型/短样本先检验逻辑，再检验 BF16。

### P4：冒烟与 pilot

- 先 20～50 个 optimizer updates，确认 loss、梯度、checkpoint、评测链路。
- 用同一 2M targets 跑 L/R/G，text_validation 和 C-Eval val 做诊断。
- 初始配置固定；必要时仅比较两档 adapter LR（1e-4、3e-4），同时对 R/G 应用，不只给 G 更多搜索预算。
- 若无明显学习，先排查 gate 饱和、target shift、命中率和新增参数冻结，再扩大预算。
- pilot 不用 C-Eval test/CMMLU test 做选择；表可以来自 train_full，这一事实必须披露，主实验所有组保持一致。

### P5：正式训练

配置冻结后，L/R/G 每组 seeds 42/43/44，各约 50M targets，全部从共同 Base 起点重新开始。预算不足时先做 seed 42 三组，结果标为单 seed 初步结果，不能包装为稳定提升。

统一保存每约 5M targets checkpoint 及最后 checkpoint；优化器/调度器/RNG/数据游标一起保存以支持真实续训。

### P6：最终评测与判断

评测 B0 与全部正式 checkpoint 的 C-Eval test/CMMLU test，生成统一报告。

- G>L 且 G>R，多个 seed 和学科一致：支持教师记忆具有增量价值。
- G≈R>L：主要支持额外记忆容量或 key 特征有用，尚未证明教师表示更好。
- G>L 但 G≈R：不能称为教师知识迁移成功。
- text loss 降低但考试分数无提升：检查训练分布与考试任务差异，不以 loss 替代主结果。
- G 只在一项基准提高：报告局部收益和另一项退化，不隐藏。
- 零/负结果且代码检查通过：记录预算和限制，再决定是否扩大语料或变更层位。

进入第二轮前再选择单一方向：表规模、教师层位、学生层位或继续训练规模；不要同时改变全部变量。

## 9. 下一步代码与配置接口

以下保留 v0.2 建议接口作为设计参考；实际已实现路径及运行命令以 [README.md](../README.md) 为准，评测入口为 scripts/evaluate.py（lm-evaluation-harness）。

```text
configs/
  experiment_v02_base.yaml
scripts/
  prepare_corpus.py       # 下载元数据/抽样/过滤/划分/tokenize
  build_keys.py           # 学生 token n-gram 精确统计
  build_memory.py         # 教师离线抽取、随机表生成
  train_clm.py            # L/R/G 共用 Trainer 与数据管道
  evaluate_mcq.py         # C-Eval/CMMLU 固定 continuation 评分
  benchmark_latency.py   # 正确 cache 实现后的性能测量
  summarize_results.py   # 汇总、paired bootstrap、学科报告
src/
  memory_lookup.py
  memory_adapter.py
  modeling_qwen3_graft.py
  data_collator.py
tests/
  test_lookup.py
  test_causality.py
  test_loss_and_gradients.py
  test_save_reload.py
  test_mcq_scoring.py
```

建议统一配置字段（设计示例，不能直接传给原版 run_clm.py）：

```yaml
experiment_version: v0.2-base
variant: teacher_memory   # lora | random_memory | teacher_memory
seed: 42
student_model: Qwen/Qwen3-0.6B-Base
student_revision: REQUIRED
student_checkpoint: REQUIRED   # 与上述模型及 revision 一致的本地完整权重路径
student_manifest: REQUIRED
teacher_model: Qwen/Qwen3-8B-Base
teacher_revision: REQUIRED
data_manifest: REQUIRED
memory_manifest: REQUIRED
memory:
  ngram_orders: [2, 3, 4]
  entries_per_order: 20000
  teacher_block_ordinal: 12
  student_block_ordinal: 3
  matching: longest_exact_suffix
  frozen: true
  fallback: zero
  short_conv: false
  gate_bias_init: -2.0
  residual_scale_init: 0.001
training:
  sequence_length: 1024
  cross_document_packing: false
  target_token_budget: 50000000
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.0
  lora_lr: 0.00002
  adapter_lr: 0.0001
  warmup_ratio: 0.05
evaluation:
  protocol: zh_mcq_base_0shot_plain_letter_loglikelihood_v2
  apply_chat_template: false
  add_special_tokens: false
  answer_prefix: "答案："
  selection_split: text_validation
  pilot_benchmark: ceval_val
  final_benchmarks: [ceval_test, cmmlu_test]
  max_prompt_tokens: 4096
  primary_aggregate: subject_macro_accuracy
```

实现时 `REQUIRED` 必须解析为真实路径/revision，缺失即报错。正式命令和可运行 YAML 由下一步代码落地后补充，并先用 smoke 验证。

## 10. 结果与可复现性

每次运行使用独立 run_id，例如 `v02base_G_s42`，保存：

```text
runs/<run_id>/
  resolved_config.yaml
  environment.lock.json
  model_manifest.json
  data_manifest.json
  memory_manifest.json
  train_log.jsonl
  checkpoints/
  evaluation/
    ceval_test_predictions.jsonl
    cmmlu_test_predictions.jsonl
    metrics.json
  report.md
```

逐题记录：benchmark、split、question_id、subject、prompt_hash、候选字符串与 token IDs、四候选 logprob、预测、正确答案、命中统计、checkpoint ID。日志中不需要复制训练文档全文或所有层激活。

主报告模板：

| 组别 | targets | trainable params | C-Eval macro/micro | CMMLU macro/micro | GPU-hours | 额外表存储 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 0 | 待测 | 待测 | 0 | 0 |
| L | 待记录 | 待记录 | 待测 | 待测 | 待记录 | 0 |
| R | 待记录 | 待记录 | 待测 | 待测 | 待记录 | 待记录 |
| G | 待记录 | 待记录 | 待测 | 待测 | 待记录 | 待记录 |

另附三 seed 统计、学科明细、差值置信区间、纠错/退化、中文验证 loss/PPL、延迟单独表和失败运行列表。未测项目写“未测”，不写成 0。

## 11. 开源参考与适用边界

1. [Memory Grafting 原论文](https://arxiv.org/html/2605.20948v1)：提供冻结教师表示、精确后缀检索及融合方法。我们的后续训练设置与论文从头预训练不同。
2. [Ultra-FineWeb 数据卡](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)：中文原料来源；需核实所锁版本字段与 split。
3. [Transformers run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)：复用纯文本 tokenization、CLM 训练和 Trainer 组织；原版 group_texts 拼接需要按本方案改造边界。
4. [DeepSeek Engram](https://github.com/deepseek-ai/Engram)：参考核心门控/查表，官方 demo 的模拟主干不能直接用于真实 Qwen 训练。
5. [C-Eval 官方仓库](https://github.com/hkust-nlp/ceval)：固定题目、split 与官方学科信息。
6. [CMMLU 官方仓库](https://github.com/haonan-li/CMMLU)：固定 dev/test、学科与答案。
7. [LLaMA-Factory 预训练文档](https://llamafactory.readthedocs.io/en/latest/advanced/trainers.html)：普通继续训练基线参考；主对照最终统一到本项目代码。
8. [MeKi 官方仓库](https://github.com/ningding-o/MeKi)：记忆模块与训练组织参考，并非本实验的教师构表方法。
9. [Qwen3-0.6B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-Base)：学生 Base 权重入口。
10. [Qwen3-8B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B-Base)：离线教师 Base 权重入口。

参考链接是资料入口，实际实施应在 environment/model/data manifest 中锁定 commit；链接到 main 不代表实验允许自动更新依赖。

## 12. 下一步启动条件

- 必需：锁定 Qwen3-0.6B-Base 与 Qwen3-8B-Base 的 revision、tokenizer 和本地完整权重路径。
- 必需：锁定教师、中文数据和两个 benchmark 的 revision/schema。
- 必需：确认实际运行机器的空闲 GPU、磁盘与隔离环境。
- 随后实现 P1～P3，用 P4 验证完整链路，再决定执行 P5 正式训练。

完成本文件仅代表实验方案已整理，不代表任何模型、数据、脚本或结果已经准备完成。

[1]: https://arxiv.org/html/2605.20948v1
[2]: https://huggingface.co/datasets/openbmb/Ultra-FineWeb
[5]: https://github.com/hkust-nlp/ceval
[6]: https://github.com/haonan-li/CMMLU
