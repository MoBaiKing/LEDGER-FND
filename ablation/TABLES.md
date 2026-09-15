# 完整消融 Tables

此表由 `python -m ablation.tables` 从当前 registry 生成。性能值仅来自实际训练，未运行不填数值。

## Table 1：Full基线与九个消融实验的真实模块设置

所有行保留原始 T/V/E/X 编码。D 表示 Direct 候选，L 表示 Deliberation 候选；U 为显式关系不确定性。

| 编号 | Variant | Evaluator | D | L | U | Minority | Global摘要 | 混合/整体分类 | 原始均值旁路 | 最终校正 |
|---|---|---|---|---|---|---|---|---|---|---|
| 基线 | full | 共享Qwen | ✓ | ✓ | ✓ | ✓ | Global token | 动态λ | ✓ | ✓ |
| exp1 | mlp_classifier | 不执行 | — | — | — | — | — | 整体MLP分类 | — | — |
| exp2 | non_llm_evaluator | 随机Transformer | ✓ | ✓ | ✓ | ✓ | Global token | 动态λ | ✓ | ✓ |
| exp3 | no_uncertainty | 共享Qwen | ✓ | ✓ | — | ✓ | Global token | 动态λ | ✓ | ✓ |
| exp4 | no_critical_minority | 共享Qwen | ✓ | ✓ | ✓ | — | Global token | 动态λ | ✓ | ✓ |
| exp5 | no_global_token | 共享Qwen | ✓ | ✓ | ✓ | ✓ | 有效pair均值 | 动态λ | ✓ | ✓ |
| exp6 | fixed_mixture | 共享Qwen | ✓ | ✓ | ✓ | ✓ | Global token | λ=.5 | ✓ | ✓ |
| exp7 | direct_only | 不执行 | ✓ | — | — | — | — | Direct | — | — |
| exp8 | deliberation_only | 共享Qwen | — | ✓ | ✓ | ✓ | Global token | Deliberation | — | ✓ |
| exp9 | no_decision_correction | 共享Qwen | ✓ | ✓ | ✓ | ✓ | Global token | 动态λ | ✓ | — |

λ为Deliberation系数：`(1−λ)D+λL`。strict Direct 的 confidence 槽位置0；它与原Full中的Direct不完全相同。

## Table 2：依赖变化与损失

| Variant | 连带变化 | 因依赖移除而关闭的loss |
|---|---|---|
| full | 完全沿用原实现 | 无 |
| mlp_classifier | concat([T,V,E,X])→LN→Linear→GELU→Dropout→Linear(2)，整体替换原分类模块 | causal_probe_classification, causal_fidelity, evidential_regularization, uncertainty_calibration |
| non_llm_evaluator | 结构、容量和预训练条件同时变化；Qwen仍用于文本编码 | 无 |
| no_uncertainty | 关系alpha/p及监督保留；u源头置0，乘法(1−u)变1 | 无 |
| no_critical_minority | 不再执行minority head，adjudication重新对有效证据softmax | 无 |
| no_global_token | G实际删除，pair槽位4..9不变；保留原global输出head接收masked mean | 无 |
| fixed_mixture | 只替换最终混合系数；内部注意力、U和校正不变 | 无 |
| direct_only | 不执行evaluator；latent confidence固定0，raw-mean旁路及IURD关闭 | evidential_regularization, uncertainty_calibration |
| deliberation_only | 不执行Direct scorer；raw-mean最终旁路关闭；IURD只使用其原有输入 | 无 |
| no_decision_correction | 直接preliminary logits；保留final_norm/classifier，跳过calibrator内部LN/残差/温度 | uncertainty_calibration |

当前三个数据集classification=1，辅助loss均为0。表中关闭项即使将来base配置启用，也会显式禁用并记录原因。

## Table 3：公平路径对照组

| 配置 | Direct | Deliberation | Direct confidence | 均值旁路 | 校正 | 是否独立训练 |
|---|---|---|---|---|---|---|
| path_direct_reference | ✓ | — | 固定0 | — | — | 与direct_only等价，复用 |
| path_deliberation_reference | — | ✓ | 不适用 | — | — | 是 |
| dual_path_reference | ✓ | ✓ | 固定0 | — | — | 是；不等于Full |

三者共用fusion_norm→final_norm→classifier。主表下游不完全一致，不把Full优势单独写成纯路径互补证据。

## Table 4：正式性能表（当前尚未运行）

当前训练源seed数量：weibo21=4、weibo=4、gossipcop=4；各数据集按实际数量生成任务。`—`表示未测量，不表示0。

| Variant | Weibo21 Macro-F1 | Weibo Macro-F1 | GossipCop Macro-F1 | 实测状态 |
|---|---|---|---|---|
| full | — | — | — | 本消融系统尚未正式训练 |
| mlp_classifier | — | — | — | 本消融系统尚未正式训练 |
| non_llm_evaluator | — | — | — | 本消融系统尚未正式训练 |
| no_uncertainty | — | — | — | 本消融系统尚未正式训练 |
| no_critical_minority | — | — | — | 本消融系统尚未正式训练 |
| no_global_token | — | — | — | 本消融系统尚未正式训练 |
| fixed_mixture | — | — | — | 本消融系统尚未正式训练 |
| direct_only | — | — | — | 本消融系统尚未正式训练 |
| deliberation_only | — | — | — | 本消融系统尚未正式训练 |
| no_decision_correction | — | — | — | 本消融系统尚未正式训练 |

正式运行后，`ablation.summarize` 会生成每数据集的完整Macro-F1/Accuracy/Fake与Real P/R/F1/AUC/NLL/Brier/ECE-15表，
并附逐seed、mean±sample std、Full同seed差值（原始小数及百分点）、困难子集样本数/类别分布、成本和失败清单。

## Table 5：成本记录口径

| 项目 | 来源 / 当前状态 |
|---|---|
| 总参数、可训练参数 | 每次实际构造模型后写入config，当前未加载完整7B模型测量 |
| LG-LED/替代Transformer/适配层参数 | 模块构造后独立计数，见head_parameter_counts.csv |
| evaluator token数 | Full 11；no_global_token 10；整体MLP分类/direct 0 |
| peak CUDA memory、wall time | 正式run的runtime_rank*.json；CPU测试不能代替7B成本结论 |
| FLOPs | 未测量；不能从参数数直接推断 |

没有任何表格预设Full最佳或预设某项下降幅度。解释边界及实际文件/函数映射见AUDIT.md。
