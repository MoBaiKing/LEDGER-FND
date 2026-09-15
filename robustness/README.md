# Robustness under Input Corruption

仅实现 **Visual Corruption: Gaussian Noise** 和 **Textual Corruption: Random
Typo Token Injection**。所有新增功能位于本目录，原 `train.py`、`evaluate.py`、
`mmfnd/`、配置和数据文件都不需要修改。

## 仓库审计与接入位置

| 环节 | 原实现与本次接入 |
| --- | --- |
| 数据 | `mmfnd/factory.py:build_loader` 读取绑定 manifest 的 `test.jsonl`；`mmfnd/data.py:CUTEFNDMultimodalDataset` 保留记录 ID、文本、标签、图片列表和 max_images。 |
| 文本清理 | manifest 准备阶段已有 HTML unescape、NFKC、零宽字符清理、空白合并；运行时不重新清理，原 collate 拼接 `text_instruction + text` 后调用原 tokenizer。 |
| 图像加载 | 原 `load_preprocessed_image` 做 EXIF 校正、透明背景合成、RGB 转换；test 不做训练 crop/brightness/contrast/JPEG。不存在的图仍按原逻辑报错。 |
| 图像 processor | 当前本地 `SiglipImageProcessor` 使用 bicubic resize 到 224×224、乘 1/255、mean/std=[0.5,0.5,0.5] normalization。 |
| Gaussian | 第一次调用原 processor，只关闭 normalization；在返回的 `[0,1]` CHW float tensor 上 `clamp(I + sigma * epsilon, 0, 1)`；第二次调用相同 processor，只关闭 resize/rescale，保留原 normalization。没有 PIL 重量化，也没有对 normalized tensor 直接加噪。 |
| Typo | 原 collate/tokenizer 后、`mmfnd/model.py:MultimodalIntrinsicEvidenceEncoder.forward` 的 Qwen 调用前。修改 `text_input_ids` / `text_attention_mask`，图像张量完全沿用原结果。 |
| Qwen 输入 | 当前 forward 只传 input_ids / attention_mask；Qwen 自动生成位置编码，不存在需要手动更新的 token_type_ids / position_ids。扩展工具遇到未知 token 字段明确拒绝。 |
| 评估入口 | 原入口是根目录 `evaluate.py`。新增 `python -m robustness.evaluate` 只编排实验，直接调用原 `mmfnd.engine.evaluate`，没有第二套模型 forward/指标实现。 |
| checkpoint | 原 `load_checkpoint` 检查 architecture/contract，支持 trainable_only checkpoint、LoRA 和共享预训练 backbone。完全不保存或修改 checkpoint。 |
| 标签及阈值 | Fake=0、Real=1；AUC 使用配置的 Fake 连续概率。每个 checkpoint 自带的 validation Macro-F1 阈值对所有 severity 冻结。test 不调阈值。 |
| 多 seed | 原 `<group>_seed<seed>/checkpoints/final_averaged.pth`；支持原默认 42,3407,2024,2025,200408，也支持任意实际 seed 列表、路径模板或重复 --checkpoint。 |

`build_robustness_loader` 只允许 test。非零 Gaussian 通过局部浅拷贝 dataset
包装 image processor，仍调用原 collate；Typo 在原 collate 返回后插入。
`none`、sigma=0、rate=0 都直接返回原 loader/collate 路径，不增加 tensor 运算。
普通训练、验证和普通测试均不会导入此模块。

## 精确定义

- Gaussian sigma: `0.00, 0.05, 0.10, 0.20`。每张图片独立，图片 shape、图片顺序、
  image_owner、文本和标签均保持原规则。
- Typo rate: `0.00, 0.05, 0.10, 0.20`。在**原 tokenizer 已截断的 prompted sequence**
  中统计有效 non-special token；包括原 instruction 的普通 token，不重新拼接或二次分词。
  数量为 `floor(rate * N + 0.5)`，因此很短的序列可能产生 0 次插入。
- 从合法 `get_vocab()` ID 集合中排除全部 `all_special_ids`，不从 embedding size
  或假设连续的 vocabulary 范围采样。PAD/BOS/EOS/CLS/SEP/MASK/UNK/其他 special
  均不能成为被选择的位置或注入的 token。
- 无放回选择原位置，在其**后面**插入一个随机 non-special token；没有 replacement。
  长度不超限时原 token 全部保留。超限时沿 tokenizer.truncation_side 截断普通 token，
  保护原 special tokens（如 BERT 的终止 SEP），再用原 tokenizer.pad 重建 padding/mask。
  当前 Qwen 使用 right padding/right truncation，无自动 BOS/EOS 模板；普通新闻超长时
  就是保留插入后序列的前 max_length 个 token。特殊模板保留规则不改变 clean 路径。
- 区分 `requested_insertions` 与截断后的 `retained_insertions`；达到 max_length
  时最终长度不再增长，也可能截掉部分插入 token。不能把长度不增长误判为 replacement。
- SHA-256 派生本地 NumPy PCG64 RNG，键包括 corruption_seed、类型、dataset、sample ID；
  图像额外含图片序号与路径。test ID 必须唯一。与 worker/batch 顺序、Python hash、全局 RNG 无关。
- 同一样本不同 severity 共用基础噪声/位置排列，使 Gaussian 强度、Typo 选中位置数量可配对比较；
  改 seed 会改变 pattern。中文/英文使用同一实现，不引入 NLP 包。
- 一次 inference 只允许一个 corruption。`--suite` 表示顺序执行两组独立实验，
  不表示联合污染。相互冲突的参数报错。

## 环境

复用仓库要求的 PyTorch/Transformers/PEFT 环境，不安装新依赖。
当前机器可用环境为：

```bash
cd /data/dyl/sotamodelv3
export PYTHON_BIN=/data/dyl/sotamodelv5/.venv/bin/python
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
```

在其他机器把 PYTHON_BIN 改为该仓库 `.venv/bin/python` 或兼容环境。
已有本地 Qwen2.5-7B / SigLIP 权重是必需的。

## 单 checkpoint：完整两组曲线

当前机器上可用于新评估协议的一个真实 checkpoint：

```bash
export CHECKPOINT=/data/dyl/sotamodelv3/workspaces/weibo21/runs/training/v3_random4_20260914_014500_weibo21_seed1933686559/checkpoints/final_averaged.pth
CUDA_VISIBLE_DEVICES=0 bash robustness/run_robustness.sh \
  --dataset weibo21 \
  --checkpoint "$CHECKPOINT" \
  --corruption-seed 2027 \
  --output-dir robustness/robustness_results/weibo21_single
```

脚本默认继承 **该 checkpoint 的 config 和 seed**，包括 max_length、max_images、
batch size、precision、manifest。不会误用当前 configs 文件中的另一个训练 seed。
`--config` 可显式指定配置（路径相对仓库根目录），`--manifest-dir` 可覆盖清单位置。
可覆盖 `--batch-size` / `--per-gpu-batch-size` 和 `--num-workers`；比较 clean 与
corrupted logits/metrics 时应保持运行配置、batch composition、设备、precision 相同。
corruption pattern 可跨 batch size 复现，不宣称任意 GPU/batching 的模型计算逐位相同。

支持环境变量：

```bash
DATASET=weibo21 CHECKPOINT="$CHECKPOINT" CORRUPTION_SEED=2027 \
  bash robustness/run_robustness.sh \
  --output-dir robustness/robustness_results/weibo21_env
```

每个输出目录是一次独立完整实验，必须为空，避免覆盖原结果或混入旧 seed/severity。
省略 `--output-dir` 时自动新建带微秒时间戳的目录。

## 单 checkpoint：单个 severity

```bash
"$PYTHON_BIN" -m robustness.evaluate \
  --dataset weibo21 --checkpoint "$CHECKPOINT" \
  --robustness gaussian --gaussian-sigma 0.10 --corruption-seed 2027 \
  --output-dir robustness/robustness_results/weibo21_sigma010

"$PYTHON_BIN" -m robustness.evaluate \
  --dataset weibo21 --checkpoint "$CHECKPOINT" \
  --robustness typo --typo-rate 0.10 --corruption-seed 2027 \
  --output-dir robustness/robustness_results/weibo21_typo010
```

两条命令各自包含一次 clean，用于计算该 checkpoint 的配对 drop。
全曲线优先用一键脚本，避免多个进程重复加载模型和计算 clean。
原 `python evaluate.py --config ... --checkpoint ...` 命令保持原行为；
新增 robustness 参数只用于 `robustness.evaluate`，不要传给根目录旧入口。

## 5-seed checkpoint

对已经训练完成的原 run_5seeds group：

```bash
CUDA_VISIBLE_DEVICES=0 bash robustness/run_robustness.sh \
  --dataset weibo21 \
  --group YOUR_COMPLETED_5SEED_GROUP \
  --seeds 42,3407,2024,2025,200408 \
  --corruption-seed 2027 \
  --output-dir robustness/robustness_results/weibo21_five_seeds
```

等价路径模板（把 group 名换成真实目录名）：

```bash
CUDA_VISIBLE_DEVICES=0 bash robustness/run_robustness.sh \
  --dataset weibo21 \
  --checkpoint-template 'workspaces/weibo21/runs/training/YOUR_COMPLETED_5SEED_GROUP_seed{seed}/checkpoints/final_averaged.pth' \
  --seeds 42,3407,2024,2025,200408 \
  --corruption-seed 2027 \
  --output-dir robustness/robustness_results/weibo21_five_seeds_template
```

也可传五个 `--checkpoint /absolute/path.pth`。seed 从 checkpoint 内读取；
同 seed 的多个 checkpoint 不算独立实验，会被拒绝。每个 severity 评估全部 seeds，
逐 checkpoint 计算 drop，再汇总 mean 和 sample std (n−1)。不会先平均阈值，
不会跨 seed 共用 clean。单 checkpoint 的 std 为 null/CSV 空值，因为样本标准差未定义。

实际近期运行使用 4 个随机 seed 时，直接继承其 seed 列表即可，无需硬凑 5 个：

```bash
bash robustness/run_robustness.sh \
  --dataset weibo21 --group v3_random4_20260914_014500_weibo21 \
  --seeds 1933686559,395953986,391173430,1728520375 \
  --corruption-seed 2027 --check-only \
  --output-dir robustness/robustness_results/weibo21_random4_preflight
```

`--check-only` 只检查所有 checkpoint 的验证阈值来源、配置一致性和 manifest，
生成 run_manifest，不进行模型 inference。正式运行请使用另一个空输出目录并移除该参数。

**旧 checkpoint**：原评估协议已经要求 validation Macro-F1 阈值。没有该元数据的
历史 checkpoint 会明确报错，不自动用 0.5，不在 robustness 中重新校准。
如需迁移，使用原根目录入口 `evaluate.py --recalibrate-on-val` 对干净 validation
校准并另存 checkpoint，再将另存的 checkpoint 显式传给本工具。
旧 clean 数字/原始 checkpoint 不会被覆盖，不应与新协议混合汇总。

## 输出

```text
<output>/
  run_manifest.json              # 参数、checkpoint/manifest SHA-256、config、阈值来源
  completion.json                # 全部成功才出现
  robustness_summary.csv         # 每 seed/severity 一行；clean 每 seed 一行
  robustness_aggregate.csv       # 每 severity 的 mean/std，包括 paired drops 和诊断
  robustness_aggregate.json
  weibo21/
    [seed_<seed>/]               # 多 checkpoint 时增加一级
      clean.json
      gaussian/sigma_0.00.json   # 明确引用同一 clean，不重复 inference
      gaussian/sigma_0.05.json
      gaussian/sigma_0.10.json
      gaussian/sigma_0.20.json
      typo/rate_0.00.json
      typo/rate_0.05.json
      typo/rate_0.10.json
      typo/rate_0.20.json
      clean_equivalence.json    # --verify-clean-equivalence 时
  figures/
    gaussian_robustness.pdf
    gaussian_robustness.png
    typo_robustness.pdf
    typo_robustness.png
```

JSON 顶层保留 accuracy/macro_f1/auc/nll/brier/ece、dataset/checkpoint/seed/severity，
并在 `metrics` 下保存原 evaluator 的完整指标和协议；`diagnostics` 保留原 evaluator
已有的 uncertainty、relation uncertainty、disagreement、routing gate、deviation、
minority、direct/deliberative/final evidence weights 均值，不修改 model 接口。

`f1_drop = clean_macro_f1 - corrupted_macro_f1`；`relative_f1_drop_pct =
f1_drop / clean_macro_f1 * 100`；Accuracy/AUC 同理。负 drop 保留。clean=0
时相对 drop 未定义，输出 null。单类别 AUC 保留原 evaluator 的未定义语义，
JSON 用 null（`undefined_metrics` 明确列出），CSV 用空值，不伪造 0。

加 `--save-predictions` 可额外保存原 evaluator 的逐样本预测/诊断 JSONL；
`text` 字段仍是原 manifest 文本，注入 token 只在运行时存在。

## 绘图与 baseline

```bash
"$PYTHON_BIN" robustness/plot_robustness.py \
  --csv robustness/robustness_results/weibo21_five_seeds/robustness_summary.csv \
  --output-dir robustness/robustness_results/weibo21_five_seeds/figures \
  --uncertainty errorbar
```

`--uncertainty shade|errorbar|none`，默认 shaded sample std。
`--csv` 可以接受多个具有同样列格式、不同 `method` 的 baseline CSV。
重复的 method/seed/severity 或不完整 seed cohort 会报错；不同 dataset 自动分目录。
不要把相同 LEDGER clean 重复放进多个输入 CSV。绘图默认拒绝诊断子集，
显式 `--allow-subset` 才能绘制带 diagnostic subset 标题的子集图。

## Correctness tests

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s robustness/tests -v
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s tests -v

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m robustness.verify \
  --config configs/datasets/weibo21.json \
  --corruption-seed 2027 \
  --output-dir robustness/test_artifacts/preprocessing

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m robustness.evaluate \
  --checkpoint "$CHECKPOINT" --device cpu --num-workers 0 --batch-size 1 \
  --limit-samples 4 --verify-clean-equivalence \
  --output-dir robustness/test_artifacts/real_checkpoint_cpu_repeat
```

CPU 使用 FP32 执行：原 engine 的 CPU 分支不启用 autocast，因此在内存中将
已加载模型转为 float32，避免 BF16 backbone 与 FP32 projection 的 dtype 冲突。
不会写回 checkpoint；该检查不等于 GPU BF16 全量论文结果。
`--limit-samples` 只切片内存中的 test records，输出明确标记 `first_N_test_samples`。
正常论文评估不传此参数。

已执行的结果及示例见 [VALIDATION.md](VALIDATION.md)。

## 文件职责

| 新增文件 | 用途 |
| --- | --- |
| `corruptions.py` | 参数互斥、stateless RNG、GaussianNoiseCorruption、RandomTypoTokenInjection |
| `pipeline.py` | 对原 test loader/collate 的局部包装、processor 分阶段调用 |
| `evaluate.py` | checkpoint/config/manifest 校验、单点/全曲线/多 seed 编排、clean 复用 |
| `results.py` | 保存原指标、配对 drop、诊断、CSV 和 mean/sample std |
| `run_robustness.sh` | 一键执行两组 severity 并绘图，继承命令行和环境变量 |
| `plot_robustness.py` | 支持 method/baseline、多 seed std、独立 PDF/PNG 曲线 |
| `verify.py` | 真实预处理示例、Gaussian 强度检查、token 插入记录、clean 输出比较 |
| `tests/test_robustness.py` | CPU fixture 的 RNG/边界/原 pipeline/5-seed/绘图集成测试 |
| `__init__.py` | 独立模块入口 |
| `.gitignore` | 不跟踪运行结果、测试图片和日志 |
| `README.md` / `VALIDATION.md` | 审计、完整命令、输出说明与实际测试证据 |
