"""Render complete exploratory matrix and paired intervals without selecting a winner."""
import json
from pathlib import Path

folder=Path('runs/layer_study')
rows=json.loads((folder/'summary.json').read_text())
baseline=json.loads((folder/'L.json').read_text())
out=['# 教师提取层与学生注入位置：实验结果','',
     '固定 seed 42、每组 2M target tokens、747 次更新、alpha=0.001；lm-evaluation-harness 0.4.13，C-Eval val 0-shot，1346 题。T/S 层号从 1 开始，注入在对应学生 block 之后。',
     '',f"LoRA 基线 accuracy：{100*baseline['groups']['graft_ceval-valid']['acc,none']:.4f}%；学科宏平均：{100*baseline['subject_macro_accuracy']['graft_ceval-valid']:.4f}%。",
     '', '22 组新增训练，复用配置一致的 T12→S3 G/R 两组及 L 基线。各教师层均有独立构建的匹配随机表。']
for title,metric in [('教师表 G 准确率（%）',lambda r:100*r['G']['accuracy']),
                     ('随机表 R 准确率（%）',lambda r:100*r['R']['accuracy']),
                     ('G−R 学科宏平均差（百分点）',lambda r:r['vs_R']['subject_macro_delta_pp'])]:
    out+=['','## '+title,'','| 教师层 / 学生位置 | S1 | S3 | S6 | S12 |','|---|---:|---:|---:|---:|']
    for t in (4,8,12):
        cells=[next(r for r in rows if r['teacher_block']==t and r['student_block']==s) for s in (1,3,6,12)]
        out.append(f'| T{t} | '+' | '.join(f'{metric(r):.4f}' for r in cells)+' |')
out+=['','## 配对区间','','| 组合 | G−R 宏平均差 [95% CI] pp | G−L 宏平均差 [95% CI] pp |','|---|---|---|']
for r in rows:
    def cell(key):
        d=r[key];lo,hi=d['subject_macro_delta_95ci_pp']
        return f"{d['subject_macro_delta_pp']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    out.append(f"| T{r['teacher_block']}→S{r['student_block']} | {cell('vs_R')} | {cell('vs_L')} |")
out+=['','## 完整验证 loss 与注入强度','','注入统计取第 741 次更新前最后一个 micro-batch，所有新旧复用训练均使用相同 batch 顺序；这是固定批次诊断，不是全验证集平均。',
      '', '| 组合 | G loss | R loss | G gate 均值 | G 实际 delta/h (%) |','|---|---:|---:|---:|---:|']
for r in rows:
    t,s=r['teacher_block'],r['student_block'];records={}
    for g in ('G','R'):
        run=Path(f'runs/gate_{g}_a001') if (t,s)==(12,3) else Path(f'runs/layer_{g}_T{t}_S{s}')
        records[g]=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    measured=next(x for x in records['G'] if x['step']==741)
    out.append(f"| T{t}→S{s} | {records['G'][-1]['validation_loss']:.7f} | {records['R'][-1]['validation_loss']:.7f} | {measured['gate_mean']:.4f} | {100*measured['effective_delta_ratio_mean']:.4f} |")
out+=['','## 解释边界','',
      '区间采用 10,000 次按学科配对 bootstrap，反映题目抽样波动，不覆盖训练 seed 方差；未作多重比较校正。12 个组合是在同一验证集上探索，最高分不构成独立验证。',
      '', '固定 alpha 不等于固定注入范数。位置变化也改变 hidden state 尺度，因此结果不能直接证明表征层级匹配；需要结合实际注入比例，以及后续尺度控制消融。',
      '', '本轮未使用 C-Eval test、CMMLU test。新 checkpoint 自动记录并恢复学生位置与记忆目录；默认旧 checkpoint 仍为 T12→S3。',
      '', '原始报告、逐题输出和配对结果在 runs/layer_study，训练曲线和 checkpoint 在 runs/layer_*。']
Path('LAYER_RESULTS.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
