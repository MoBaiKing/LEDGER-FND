# sotamodelv3 可复现消融系统 v2

完整模型审计见 [AUDIT.md](AUDIT.md)，完整配置/结果表见 [TABLES.md](TABLES.md)，检查记录见 [VALIDATION.md](VALIDATION.md)。
研究结论仅来自实测，配置不含预设性能排名或下降幅度。

本目录只复用当前 v3 架构，正式训练/模型/评估文件未改动。
保留 `exp1`–`exp9` 九个连续消融编号；Full作为独立比较基线，不占用消融编号。
命令、输出和汇总统一使用明确的 variant 名称。
旧 v1 checkpoint、配置和结果不能混入 v2；模型定义和 seed 来源均已变化。

## Full基线与九个消融实验

| 原编号 | variant | 真实变化及连带停用 |
|---|---|---|
| 基线 | full | 原始全部机制、loss、超参数及模型选择 |
| exp1 | mlp_classifier | 将编码后的 `[T;V;E;X]` 拼接后直接送入一个 MLP 输出二分类 logits，整体替换原分类模块（reliability、LG-LED、融合、IURD及原线性分类头均不执行） |
| exp2 | non_llm_evaluator | 随机小 Transformer 替代 latent Qwen；保持槽位/因果可见性，保留下游 |
| exp3 | no_uncertainty | u 在源头置0，所有显式使用路径中性化；保留 alpha/p/关系头及合法监督 |
| exp4 | no_critical_minority | 删除 minority head，保留常量特征槽位及其余 softmax 归一化 |
| exp5 | no_global_token | 实际删除末尾 G，10-token evaluator；用有效 pair states masked mean 进入原 global head |
| exp6 | fixed_mixture | λ固定0.5（Deliberation系数），不执行动态 gate，其余内部机制保持 |
| exp7 | direct_only | 原 direct scorer 的 latent confidence 固定0；不执行 evaluator，关闭 raw-mean 最终旁路和 IURD |
| exp8 | deliberation_only | 不执行 direct scorer；最终仅由 deliberation候选进入 fusion_norm，关闭 raw-mean 最终旁路，保留 IURD |
| exp9 | no_decision_correction | 直接使用 preliminary_logits，保留前置 final_norm/classifier 和上游全部路径 |

主表每个名称唯一，Direct+Deliberation且所有正式机制齐备就是 Full。
严格 Direct 与 Full 的下游不同，另提供公平路径组：`path_direct_reference`、
`path_deliberation_reference`、`dual_path_reference`，统一无raw-mean残差、无IURD，
Direct confidence固定0。第一项与 direct_only 等价并复用结果；其余两项独立训练。

exp1中的 MLP 是从四路编码特征到二分类 logits 的完整分类器，不再拆成“简单融合层+原分类头”。它不等价于 Direct，后者仍是原 evidence scorer 的加权分类路径。
非 LLM 对照同时改变结构、容量和预训练，不单独证明LLM推理能力或排除参数量影响。
Global 对照是 token 汇总与 masked pair mean 的比较，不是去掉全部全局信息。
Minority 删除实验不能单独排除容量带来的收益。

## 实际运行命令

在项目环境中运行；依赖版本沿用原 `requirements.txt`，不改变 CUDA/Conda。
本消融系统新增的隔离测试依赖统一存放在 `ablation/.runtime/`，pip、Python、Hugging Face和Torch缓存统一存放在 `ablation/.cache/`；通过 `ablation/python_env.sh` 调用，避免写入系统环境或 `/tmp`。
当前正式训练是4个随机seed，不是固定5个。所有命令通过训练计划导入实际 seed 集合及运行超参数。
以后计划有5个seed时自动运行5个；当前不会补造第5个。默认顺序执行不同variant，单次内部可DDP。

```bash
cd /data/dyl/sotamodelv3

# 仅生成计划和完整配置，不加载模型或训练。
python -m ablation.run --datasets weibo21 weibo gossipcop \
  --training-plan workspaces/scheduled/v3_random4_20260914_014500/plan.json \
  --suite main_v2 --with-path-controls --dry-run

# Full 小规模检查，单独输出，不能参与科研结果汇总。
python -m ablation.run --dataset weibo21 --variants full \
  --suite smoke_v2 --smoke-steps 2 --execute

# 单项：导入当前训练seed，而不是产生新的随机seed。
python -m ablation.run --dataset weibo21 --variants no_uncertainty \
  --suite main_v2 --execute

# 单数据集全部主消融+公平路径组。
python -m ablation.run --dataset weibo21 --suite main_v2 \
  --with-path-controls --execute

# 三数据集完整套件：当前10×4×3主实验，加2×4×3非重复路径对照，共144次训练。
# 当前没有自动执行本命令。指定同一计划确保与当前训练的seed、预算一致。
python -m ablation.run --datasets weibo21 weibo gossipcop \
  --training-plan workspaces/scheduled/v3_random4_20260914_014500/plan.json \
  --suite main_v2 --with-path-controls --execute

# 相同命令加 --resume 从各自 last.pth 续训，完整实验自动跳过。
python -m ablation.run --datasets weibo21 weibo gossipcop \
  --training-plan workspaces/scheduled/v3_random4_20260914_014500/plan.json \
  --suite main_v2 --with-path-controls --execute --resume

# 汇总允许不完整套件，明确显示实际完成数量及缺失/失败原因。
python -m ablation.summarize --suite-dirs \
  ablation/outputs/weibo21/main_v2 \
  ablation/outputs/weibo/main_v2 \
  ablation/outputs/gossipcop/main_v2
```

默认每次1个进程，与当前计划的单GPU训练方式一致；`--nproc-per-node 4` 可显式启用DDP，
此时有效batch为 per_gpu_batch × accumulation × 4，必须为所有variant保持同样设置。
`--base-config`、`--manifest-dir` 仅用于单数据集的共同覆盖；`--seeds` 可显式覆盖源记录，覆盖事实写入协议。
`--seed-source` 是 `--training-plan` 的别名，可读取带 `seeds` 列表的 JSON；未指定时查找当前训练计划。
非LLM结构通过 `--non-llm-config` 传 JSON，默认 hidden_dim=256/layers=2/heads=8/FFN=1024/dropout=.1。
`--fixed-mix-coefficient` 默认.5；其他值必须用新suite预先定义，禁止按测试结果挑选。

旧的正式 `train.py` 命令保持原行为。不指定消融参数的 adapter 模型默认为Full。
`expN/config.json` 是可审阅的定义及模块状态；runner 通过 registry 生成完整运行配置，不能直接把定义文件传给 train.py。

## 数据、种子和恢复

- 不重新划分现有 train/val/test。保存所有manifest的SHA256及源训练计划、Git commit、dirty状态、代码hash。
- 同一seed下公共编码器/投影/分类头初值一致；新增模块的初始化隔离RNG，不改变公共随机流。
- 每项都从相同的预训练起点独立训练，不以训练好的Full做测试时关闭模块。
- 保存解析后完整配置、配置hash、模块实际状态、模块参数数、最终阈值来源、checkpoint、epoch及逐rank耗时/显存。
- 输出为 `ablation/outputs/<dataset>/<suite>/<variant>/seed<seed>/`，正式实验目录不会被覆盖。
- 完成判据包含最终平均checkpoint、val/test指标、test预测和final_summary，已有完整run自动跳过。
- 未完成目录必须使用 `--resume` 且有可恢复的 `last.pth`，否则记录 blocked。配置/协议改变拒绝复用同suite。
- 续训恢复模型、优化器、scheduler、scaler及各rank Python/NumPy/Torch/CUDA RNG；world size必须一致。
  **多进程persistent数据worker的内部RNG没有快照**，与原训练入口一样只支持epoch边界恢复，
  不保证num_workers>0时与未中断增强序列逐位一致；要求严格恢复可预先对所有variant统一设 num_workers=0。
- 固定预训练权重和原图内容；本系统hash覆盖manifest及代码，不覆盖全部图片/7B权重字节。

## 评估及表格

Fake=0，Real=1，`fake_prob=softmax(final_logits)[:,0]`，`p>=threshold`判Fake。
AUC target为`label==0`。每个最终平均模型在自己的validation上按原网格选Macro-F1阈值，test固定使用。
模型内部IURD温度是消融机制；验证集阈值选择是全部配置共用的外部评估协议，不随消融关闭。

NLL、Brier、ECE都用最终概率。Brier指标沿用单列Fake概率定义；ECE-15是15个等宽top-label confidence bins。
完整导出Macro-F1、Accuracy、Fake/Real P/R/F1、AUC、NLL、Brier、ECE。单类AUC写null及原因。
逐样本包括sample_id、dataset、seed、variant、true/pred label、fake_prob、阈值、前后logits、证据/pair可用性。
无审议/无校正的诊断字段写null并说明原因，不把内部接口占位常数当作真实测量。

困难子集默认使用预先记录的Full参考seed（导入列表第一个）及validation分位点规则：
conflict为disagreement≥验证集75%分位点，hard为距该模型决策阈值≤验证集25%分位点。
同一套test ID用于所有variant/seed，不使用test标签或“Full对、消融错”来挑样本。
这是Full定义的诊断子集，不是人工真实冲突标签；Full参考未完成时标记待定义。
每个子集导出具体ID、阈值、样本量和类别分布，空子集不填假值。

汇总产物：

- `per_seed_metrics.csv`：每seed完整指标、成本、epoch/checkpoint。
- `mean_std.csv`：每数据集/variant/子集实际n、均值和样本标准差（n−1），n=1时std=NA。
- `paired_deltas.csv`：同seed配对，主表对Full，公平路径组对dual_path_reference。
  `delta_macro_f1=full−variant`；`delta_macro_f1_pp=100*delta_macro_f1`。
- `module_states.csv`：模块启用状态；`missing_failed.csv`：未完成/失败/不适用清单。
- `results.md` / `results.json`：可读结果表、完整追溯信息及冻结子集。

Full不保证最好；提升、下降、失败及缺失seed均原样报告，不筛掉不利结果。

## 检查命令

```bash
python -m unittest ablation.test_ablation -v
python -m unittest discover -s tests -v
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module ablation.test_ablation --ddp
```

测试使用tiny Qwen和固定合成batch，仅替代昂贵backbone与数据；真实reasoner、decision、loss、训练入口、
模型选择、checkpoint、最终评估和汇总逻辑参与执行。测试结果不写入正式性能表。
