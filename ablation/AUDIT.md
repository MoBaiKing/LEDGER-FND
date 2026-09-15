# 当前 v3 模型与消融审计

审计范围仅为 `/data/dyl/sotamodelv3`，2026-09-14。正式架构仍为
`qwen_lora_lgled_v3`，确实包含双路径、三类关系和 IURD 校正，十项主实验均适用。
本次未修改正式 `mmfnd/model.py`、`mmfnd/latent_evidence_deliberation.py`、
`mmfnd/engine.py`、`train.py`、`evaluate.py`；改动限定于 `ablation/`。

## 模块 → 实现 → 输入输出 → 开关 → 依赖

记 B=batch，D=384（配置可改），H=Qwen hidden size（从实际 config 读取），L=文本长度，
M=本批所有图片数，P=图片 patch 数。T/V/E/X 均为 `[B,D]`。

| 模块 | 正式文件 / 函数 | 实际输入 → 输出 | 正式开关 / 消融入口 | 依赖及审计结论 |
|---|---|---|---|---|
| T 文本证据 | `mmfnd/model.py::MultimodalIntrinsicEvidenceEncoder.forward` | Qwen `[B,L,H]` → `text_proj` → 最后非 padding token `[B,D]` | `qwen_*`, `lora_*` | 所有实验均保留 Qwen 文本编码；移除 evaluator 不等于移除文本 Qwen |
| V 视觉证据 | 同类 `forward/aggregate_images` | SigLIP `[M,P,Hv]` → patch mean → 按 owner 聚合 `[B,D]` | `freeze_vision`, `unfreeze_vision_last_n` | 当前多图聚合，不替换成其他历史版本视觉分支 |
| E 内在/事件证据 | `IntrinsicEventEvidence.forward` | 文本自关系、局部/全局图文对齐 → `intrinsic_feature [B,D]` | 当前模型固有，无独立 LG-LED 开关 | E 不是额外检索证据或第三个外部模型 |
| X 交互证据 | `MultimodalIntrinsicEvidenceEncoder.cross_consistency` | concat(`abs(T−V)`, `T*V`) `[B,2D]` → LN/Linear/GELU `[B,D]` | 固有 | 依赖 T、V；不能改名后改变含义 |
| 模态/证据有效性 | `UncertaintyAwareEvidenceReasoner.forward` | 三路 bool mask `[B,3]` → X 可用性 `T & V` → `[B,4]` | `ablate_component`, `view_dropout_probability` | 文本消融同时禁用 E，图像消融同时禁用 E；pair mask 为两端可用性 AND |
| reliability probe | `CausalReliabilityGate.forward` | T/V/E → 三路权重、leave-one-out effects、probe logits、reliable consensus | `causal_effect_scale` | 位于审议前，供校正及可选辅助损失使用；不是 Critical Minority |
| exp1整体MLP分类器 | `ablation/model.py::MLPClassificationModule` | concat(`[T,V,E,X]`) `[B,4D]` → LN→Linear(D)→GELU→Dropout→Linear(2) | `mlp_classifier` | 从编码特征直接输出 logits；reliability、LG-LED、融合、IURD和原线性分类头均不构造为有效分类路径 |
| 共享证据投影 | `latent_evidence_deliberation.py::_project` | 四路 `[B,4,D]` → 共享 D→H、role embedding、norm `[B,4,H]` | `projector_type`, `use_role_embedding` | 四种证据角色共享投影；exp1/strict Direct 移除该 evaluator 专用投影 |
| Pair/Global 槽位 | `LatentJudgeTokenBank`, LG-LED `forward` | evidence 0..3 + pair 4..9 + G=10 → `[B,11,H]` | `judge_type`, `use_global_judge` | Pair 固定为 TV、TE、TX、VE、VX、EX；删除末尾 G 不移动 pair 编号 |
| evaluator attention | `_run_qwen_last_layers` | 11 token → 共享 Qwen 最后 N 层 → `[B,11,H]` | `latent_judge_num_layers=2` | 标准因果可见性 j≤i，attention 输入全 1；缺失 evidence 投影清零，但正式实现并未在 attention 中屏蔽其 key |
| 非 LLM evaluator | 原 `MLPPairJudge`（本次不用） / 新 `SmallCausalEvaluator` | 相同槽位 H→h→随机 Transformer→H | `ablation.non_llm_evaluator` | 新对照保留因果可见性，使用固定正弦位置编码；结构、容量和预训练条件都改变 |
| 关系头 | `EvidentialRelationHead.forward` | pair states `[B,6,H]` → softplus evidence、alpha=e+1、p `[B,6,3]` | `use_evidential_relation` | 类别保持 **agreement / ambiguity / conflict**；无 pair 真实关系标签 |
| 关系 uncertainty | 同上及 LG-LED `forward` | u=3/sum(alpha)，`[B,6]` | `use_uncertainty`（原开关不彻底）；v2 `no_uncertainty` | 影响 certified conflict、deviation、minority、adjudication、routing、IURD；v2 在源头消除全部显式路径 |
| evidence confidence | `EvidenceConfidenceHead.forward` | latent evidence states `[B,4,H]` → `[B,4]` | 无独立开关 | 独立 learned head，不是直接把 u 重命名成 confidence；原 Direct 读取它，因此原 Direct 含 latent 信息 |
| Critical Minority | `CriticalMinorityHead.forward` | evidence state + confidence/deviation/mean u/mean conflict → 4 个标量 | `use_critical_minority` | 没有单独少数证据聚合通道；标量进入 adjudication scorer。关闭后用常量槽位，剩余证据 softmax 正常归一化 |
| Global 偏好 | `GlobalJudgeHead.forward` | hG `[B,H]` → 四路 logits `[B,4]` | `use_global_judge` | 进入 adjudication 特征；no_global_token 改用有效 pair hidden states masked mean，保留此现有输出头 |
| Deliberation Fusion | `EvidenceAdjudicationHead`, LG-LED `forward` | latent state及5个标量 → evidence softmax → 加权原始证据 `[B,D]` | 消融 `deliberation_only` | 审议用于产生权重；不是把 Qwen hidden state 直接作为分类特征 |
| Direct Fusion | `DirectFusionHead.forward` | concat(原始 evidence, latent confidence) → evidence softmax → `[B,D]` | 消融 `direct_only` | 严格 Direct 将 confidence 槽位置0，不调用 evaluator；保留原 evidence scorer 参数 |
| 混合 | LG-LED `forward` | `g=(1−λ)g_direct+λg_delib` | threshold=.3, scale=10 | λ 是 **Deliberation 系数**；`fixed_mixture` 固定 .5，不执行原动态 gate |
| 公共原始均值残差 | LG-LED `fusion_residual_scale/fusion_norm` | g+可学习scale·mean(evidence) → LN `[B,D]` | 正式无开关，v2 状态表显式记录 | strict Direct/Deliberation及公平路径组三者关闭 raw-mean 最终旁路；Full及其他局部消融保持 |
| Final Decision Correction | `UncertaintyCalibratedDecision`, `IntrinsicUncertaintyResidualDisentangler` | final_norm→共享classifier preliminary logits；uncertainty加权残差扣除→内部LN→同classifier/temperature | 消融 `no_decision_correction` | 关闭时直接返回 preliminary logits，保留前置 final_norm/classifier；不计算已停用 calibrator |
| 最终预测 | `mmfnd/engine.py::evaluate` | logits→softmax[:,0]→验证集阈值→0/1 | 统一阈值网格 | Fake=0、Real=1；`fake_prob>=threshold` 判0；不做第二次模型内校正 |
| 模型选择 | `train.py::main/average_checkpoints` | epoch val Macro-F1 → early-stop/top-k → 平均模型再做val选阈值 → 一次test | 当前训练配置 | 不更改选择指标、平均方式、类别、超参数或测试使用时点 |

## 实际缺失模态边界

正式 reasoner 默认将三路证据视为可用，仅显式模态消融/view dropout 改 mask。
正式 data collate 不支持整批没有任何图片：image processor 无空输入分支。
因此本系统不宣称新增了任意原始缺图数据支持，也不偷偷改变 Full 的 mask。
新融合/evaluator 已测试有效证据 mask：无效 evidence 被清零、pair 有效性正确、权重归一化、
只有一路可用时 Global pair mean 为0且无 NaN。原始数据必须满足原数据集 manifest 合约。

## 损失审计

| `loss` key | 正式定义 / 使用输出 | v2 处理 |
|---|---|---|
| classification | 最终 logits 的带 smoothing CE | 所有配置保留 |
| contrastive | T/V 双向 batch 对比 CE，temperature .07 | 所有配置在原权重非0时保留 |
| causal_probe_classification | reliability probe 的 CE | exp1因整体替换分类模块而关闭；其余配置有定义时保留原权重 |
| causal_fidelity | probe 对最终分类分布的 KL | exp1因整体替换分类模块而关闭；其余配置有定义时保留原权重 |
| event_counterfactual_ranking | 匹配/错配图文 cosine hinge | 保留；batch=1 时无定义的成对项为0，沿用原逻辑 |
| visual_source_ranking | 有效多图来源匹配/错配 hinge | 保留；无有效多图时不计算 |
| causal_veracity_ranking | 最终 Fake/Real logit margin 的成对损失 | 保留；批次缺一类别时不计算 |
| uncertainty_calibration | IURD uncertainty 与真类误差的 smooth-L1 + 两列 Brier loss | 无 correction 的配置关闭并记录原因；这不是评估的单列 Brier 指标 |
| evidential_regularization | relation strength mean（占位可选正则） | 无 evaluator 的 exp1/direct 关闭；no_uncertainty 保留关系头和此可选目标 |
| minority/global/gate 专属监督 | **not_applicable：正式代码不存在** | 不新增、不伪称已删除 |
| pair 关系标签监督 | **not_applicable：正式代码不存在** | 保持原关系类别，不伪造 pair 标签 |

当前三个正式配置均 classification=1，其余上述辅助权重=0。
Full 仍调用原始完整 loss 函数，包含其历史零权重项；其他变体只计算仍有效且权重非零的目标，
不通过乘0执行已删除的模块。DDP 只接收实际 loss 的可导图，其他诊断输出 detach，
配合 `find_unused_parameters=True` 正确处理未使用参数；已删除旧版全参数“零梯度锚点”。

## 双路径对照为何必须额外存在

严格 direct_only 必须关闭 latent confidence 与 IURD，原 Full 的 Direct 则读取 latent confidence，
且 IURD 依赖 relation uncertainty。因此主表的 direct_only、deliberation_only、Full 下游不一致，
不能把 Full 更好单独归因于“纯粹的双路径互补”。

公平路径组三者共同使用现有 fusion_norm→final_norm→classifier，不含 raw-mean 最终残差或 IURD。
Direct scorer 的 confidence 固定0，双路径参考中也相同；Deliberation 内部 confidence 仍保留。
`dual_path_reference` 与 Full 的差异明确保存，不冒充 Full。
`path_direct_reference` 与 strict direct_only 在参数初值、运算、loss 上完全等价，调度器记录 alias 并复用结果。
该组三种路径存在与否仍会改变容量和训练梯度，结论限于该共同下游设置。

## 当前训练 seed 来源

当前实际计划为 `workspaces/scheduled/v3_random4_20260914_014500/plan.json`。
计划读取出的 seed 为 1728520375、1933686559、395953986、391173430，数量为4。
这些数值仅是本次审计记录，**没有硬编码为消融默认值**。
runner 从指定训练计划/seed记录导入；未指定时查找当前数据集最新训练计划，并打印、保存来源及hash。
无可用计划时才读取数据集配置的单个实际 seed，不生成新seed或退回历史固定5-seed列表。
训练计划中的 config 快照及 epochs/patience/batch/accumulation 命令覆盖也会导入。
