# v2 完整检查记录（2026-09-14）

最终检查全部通过；未启动真实7B/多GPU正式训练。测试的合成batch结果不进入性能表。

| 检查 | 实际范围 | 结果 |
|---|---|---|
| 消融测试 | `python -m unittest ablation.test_ablation -v` | **16/16通过** |
| 原项目评估测试 | `python -m unittest discover -s tests -v` | **15/15通过** |
| 双进程DDP | CPU/Gloo，10主配置+3路径配置，各2个含no_sync累积的优化step | **13/13配置通过** |
| Full一致性 | 同权重/输入/RNG，FP32全部输出、loss及一步AdamW后的参数；CPU BF16最终logits | 精确相等，rtol=0/atol=0 |
| 严格路径隔离 | exp1只执行编码器+整体MLP分类器；direct不调用latent Qwen/IURD；deliberation无Direct scorer/均值旁路；Direct alias一致 | 通过 |
| 干预测试 | 扰动raw uncertainty不改变no_uncertainty logits；fixed不调用dynamic_gate | 通过 |
| Global/非LLM | G实体槽位删除，仅10token；有效pair均值；随机Transformer槽位与因果mask、未来token不可见 | 通过 |
| 数值与mask | 全部变体缺失证据权重、反向梯度、只有一路可用；CPU BF16 logits有限 | 通过 |
| 损失与梯度 | 被移除目标不求值，保留合法非零辅助项，真实loss反向与优化 | 通过，无全参数零梯度锚点 |
| 真实训练入口闭环 | 十个主配置均经真实train.py循环、top-k平均、validation阈值、final test、checkpoint及汇总 | 通过；backbone/数据由tiny Qwen及固定batch替代 |
| 中断/续训 | 含dropout的epoch1后主动中断，从last.pth恢复到epoch3，与连续训练比较 | 参数精确一致（无数据worker fixture） |
| checkpoint隔离 | 拒绝跨variant、跨seed、配置/协议不同的加载 | 通过 |
| 汇总完整性 | 每个variant仅完成1/2个fixture seed，报告真实n及缺失；类别概率/阈值/前后logits一致 | 通过 |
| 实际训练seed导入 | 当前三数据集计划中的4个随机seed，以及epochs/batch等命令覆盖 | 通过，没有固定seed列表 |
| 三数据集dry-run | 每数据集48个独立run+4个alias记录；总144条训练命令，12条复用记录 | 已生成，**没有执行** |
| 未完成性能表 | 三数据集每项均0/4；所有指标为—，失败/缺失清单及待定义子集状态明确 | 已生成 |
| 语法与CLI | compileall、train/evaluate帮助、调度与汇总CLI、diff whitespace | 通过 |
| 正式文件完整性 | model.py/latent_evidence_deliberation.py/engine.py/train.py/evaluate.py与修改前SHA256比较 | 全部一致 |

最终测试依赖位于 `/data/dyl/sotamodelv3/ablation/.runtime/python-packages`，所有消融环境的pip、Python、Hugging Face和Torch缓存位于 `/data/dyl/sotamodelv3/ablation/.cache`：
Python 3.10、PyTorch **2.4.1+cpu**、torchvision **0.19.1+cpu**、Transformers **4.46.3**、PEFT **0.13.2**。
原训练环境的PyTorch2.2.2缺少当前train.py要求的torch.amp.GradScaler；
已仅在上述消融隔离目录安装项目指定版本，并通过 `ablation/python_env.sh` 重跑全部最终检查。
没有修改原训练Conda/虚拟环境、CUDA、依赖清单或正在运行的任务。

完整日志：`verification/unit_tests.txt`、`verification/evaluation_tests.txt`、`verification/ddp_tests.txt`。
检查清单及通过状态同时保存为 `verification/checks.json`。
可审阅的实际配置/命令位于 `outputs/<dataset>/validated_v2/`。

## 明确尚未执行的检查

- 真实Qwen2.5-7B、SigLIP权重与真实数据的forward/backward。
- CUDA BF16、NCCL、多GPU显存峰值及真实吞吐/FLOPs；CPU BF16不替代GPU验证。
- 所有正式训练及研究性能比较；当前没有证据说明哪个variant优于另一个。
- persistent多worker数据增强状态的逐位恢复：当前不保存worker内部RNG，见README的恢复边界。
- 任意原始缺图数据的支持：原始collate整批无图并不受支持；本次没有改变Full数据协议。

模块参数计数只实例化CPU heads，D=384、H=3584，未加载backbone。
可用 `python -m ablation.tables --parameter-counts` 重新生成。

## 修改文件清单

- 重写：`registry.py`、`model.py`、`entrypoint.py`、`run.py`、`summarize.py`、`test_ablation.py`。
- 配置保留：`exp1`–`exp9`；Full独立作为基线，不占消融编号。
- 新增：`loss.py`、`tables.py`、`AUDIT.md`、`TABLES.md`、`module_states.csv`、`head_parameter_counts.csv`。
- 新增：`path_controls/`下三个对照定义、`verification/`检查记录。
- 生成但git忽略：`outputs/`中的完整配置、运行清单、未完成结果表。

原有正式模型、数据、checkpoint、训练脚本及此前工作区未提交修改均保留。
