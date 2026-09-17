# T12→S1 三种子确认与机制归因

本轮冻结上一轮选择，不继续搜索层位。学生为 Qwen3-0.6B-Base，教师为 Qwen3-8B-Base block 12，注入在学生 block 1 后。训练数据、500k N-gram 表、优化器和 5M-token 日程保持不变。

## 预注册比较

主比较为 full G、full S、L 的 seeds 42/43/44。CMMLU 是主指标，C-Eval validation 是次指标。seed 42 复用已完成机制实验，补训 seeds 43/44。

组件消融对 full G 做同 seed 配对比较，每组运行 seeds 42/43/44：

- `nofallback`：保留教师表和 ShortConv，只关闭 trainable hash fallback。
- `noshortconv`：保留教师表和 fallback，只关闭 ShortConv。
- `fallback_only`：所有位置走 fallback，教师表不参与 K/V，保留 ShortConv。

每个 checkpoint 先评测 CMMLU，再评测 C-Eval；非 L 组额外运行固定 100k-token adapter on/off loss 与注入强度诊断。主结论以三 seed 配对差值为单位，不能用单次最高分代替均值和方向一致性。

若 full G 在 CMMLU 上不能稳定优于 full S，或优势仍只出现在 C-Eval，下一阶段不再扩大普通 continued-pretraining 预算，转向教师/学生显式 alignment pretraining、teacher table whitening 或 contrastive projection。
