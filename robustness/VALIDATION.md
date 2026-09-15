# 实际验证记录 — 2026-09-14

新增代码全部位于 `/data/dyl/sotamodelv3/robustness`。
没有改动原 training/evaluation/model/metric/config/dataset 文件；
仓库原先已有的未提交修改保留原状。未重新训练、未保存 checkpoint。

环境：Python 3.10，PyTorch 2.4.1+cu124，Transformers 4.46.3，PEFT 0.13.2。
使用 `/data/dyl/sotamodelv5/.venv/bin/python` 复用本机依赖。

## A. Clean equivalence：真实 LEDGER checkpoint

checkpoint：

```text
/data/dyl/sotamodelv3/workspaces/weibo21/runs/training/v3_random4_20260914_014500_weibo21_seed1933686559/checkpoints/final_averaged.pth
```

实际载入本地 Qwen2.5-7B-Instruct、SigLIP、该 checkpoint 的 trainable weights；
使用原 `mmfnd.engine.evaluate`，原验证集阈值 0.54。
CPU FP32，batch size=1，num_workers=0，前 4 条真实 Weibo21 test 样本。

| 比较 | logits 最大绝对差 | probabilities 最大绝对差 | metrics |
| --- | ---: | ---: | --- |
| 原 none vs Gaussian sigma=0 | 0.0 | 0.0 | 一致 |
| 原 none vs Typo rate=0 | 0.0 | 0.0 | 一致 |

允许容差 atol=1e-7、rtol=1e-6；本次实际差为零。
对应报告：`test_artifacts/real_checkpoint_cpu/weibo21/clean_equivalence.json`。
checkpoint 文件 SHA-256 在评估前后相同。

这是同 checkpoint 同运行设置下的真实 forward 等价性检查，
不是全量测试集复现，也不是 GPU BF16 发布结果验证。
前 4 条样本均为 Fake，所以 AUC 未定义，遵循原 evaluator 的 NaN 语义，
JSON 正确保存为 null，并列入 undefined_metrics。没有人为填充 AUC。

## B. Gaussian severity：真实图片

样本 `weibo21-test-000000` 第一张图片；原 processor resize/rescale 后加噪，
corruption seed=2027。

| sigma | 像素 RMSE | pixel range | shape |
| --- | ---: | --- | --- |
| 0.05 | 0.04860391095 | [0, 1] | 3×224×224 |
| 0.10 | 0.09528238326 | [0, 1] | 3×224×224 |
| 0.20 | 0.18079942465 | [0, 1] | 3×224×224 |

强度严格增加；使用同一基础 epsilon；clamp 后的 RMSE 不要求精确等于 sigma。
重复 seed 得到完全相同像素，seed=2028 得到不同像素。

图片：`test_artifacts/preprocessing/gaussian_example.png`。
数值：`test_artifacts/preprocessing/examples.json`。

## C. Typo severity：真实 tokenizer 的 4 个中文样本

使用实际 Qwen tokenizer；下表计入原 prompted sequence 中的普通 token。

| sample ID | clean 长度 | 5% 插入数 | 10% 插入数 | 20% 插入数 | 20% 最终长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| weibo21-test-000000 | 125 | 6 | 13 | 25 | 150 |
| weibo21-test-000001 | 32 | 2 | 3 | 6 | 38 |
| weibo21-test-000002 | 32 | 2 | 3 | 6 | 38 |
| weibo21-test-000003 | 43 | 2 | 4 | 9 | 52 |

这些样本均未达到 max_length=192，原始 token 全部保留。
注入 token 不含 special ID；完整 token ID/token string 列表、插入位置、截断统计
已打印到 `preprocessing_check.log` 并保存到 `examples.json`。
同时验证了真实 tokenizer 的 attention_mask 与最终 ID 对齐。

真实局部示例（sample 000001，5%，seed 2027）：

```text
原始：... 104013, 5373,         20450, 5373, ...
插入：... 104013, 5373, 123288, 20450, 5373, ...

原始：... 33108, 111418,        105814, 1773, ...
插入：... 33108, 111418, 10257, 105814, 1773, ...
```

123288 和 10257 是该 tokenizer 的合法 non-special ID。
原始 32 tokens 变为 34 tokens，没有 replacement 或删除。
更短/空/仅 special 序列、满长右/左截断和 special 尾标记保护由 CPU 边界测试覆盖。

## D. Determinism 与集成

`python -m unittest discover -s robustness/tests -v`：**12/12 通过**。

- zero severity 保持对象/原路径，corruption 不消耗 NumPy/Torch 全局 RNG。
- Gaussian 重复相同 seed 完全一致；改变 sample/dataset/image index/seed 得到独立 pattern。
- Gaussian normalization 与手工 `(noisy_pixels - .5)/.5` 逐位一致；文本/labels/image_owner 不变。
- Typo 保留原序列、合法 vocabulary、special token 保护、round-half-up 数量正确。
- 右/左截断、padding、attention mask、空序列、拒绝未知额外 tokenizer 字段。
- 同一实现处理中文/英文 token sequence。
- 原缺失图片报错保留，源图片文件哈希不变。
- worker 0/batch 2 与 worker 2/batch 3 的逐样本有效 IDs 和图片张量完全一致，
  Gaussian/Typo 两种方法都验证。
- 真实 shared evaluator + 输入敏感的 CPU fixture + 原 checkpoint save/load 验证零噪声 logits/probs/metrics。
- 五个独立 seed 的临时 fixture checkpoint 完整执行 7 个 setting：共 35 行，
  每个 checkpoint 只做一次 clean；零 severity JSON 复用同一份指标。
- 5-seed sample std / paired drops / undefined 值，以及两张 PDF/PNG 绘图都通过。

原仓库 `python -m unittest discover -s tests -v`：**15/15 通过**。
原阈值协议、标签方向、checkpoint provenance、训练控制流和多 seed 汇总回归检查均通过。

`bash -n robustness/run_robustness.sh`、Python compileall 检查通过。

## 真实两组 severity 冒烟执行

同一真实 checkpoint、同一 4 样本 CPU 配置，执行：
clean 一次 + Gaussian 0.05/0.10/0.20 + Typo 0.05/0.10/0.20。
所有 7 个 setting 完成，真实模型可接收两类非零 corrupted inputs。
生成 7 行 per-severity CSV、9 份 clean/别名/severity JSON、aggregate CSV/JSON、
Gaussian/Typo 各一套 PDF/PNG。checkpoint 文件哈希前后保持一致。

输出：`test_artifacts/real_suite_cpu/`；完成标记为 `completion.json`。
图表标题明确包含 `diagnostic subset`，不能当作论文鲁棒性结果。

另对当前真实 4-seed group 完成无 inference 预检：
`v3_random4_20260914_014500_weibo21`，seeds
1933686559、395953986、391173430、1728520375。
全部 checkpoint/manifest/配置/阈值验证通过，报告位于
`test_artifacts/real_multiseed_preflight/run_manifest.json`。

尚未运行全量 test × 全 severity × 多 seed 的正式论文实验。
5-seed 全流程检查使用临时 fixture，不能被误解为 5 个真实 LEDGER checkpoint 的性能实验。
完整正式运行命令见 README；默认不带 limit_samples，继承每个 checkpoint 的原配置。

## 最后核对

- Gaussian 仅改变视觉输入；Typo 仅改变文本 token 输入。
- 入口/API 均限制为 test；训练和 validation 原路径不变。
- 没有模型结构、loss、checkpoint 格式或源数据修改，没有重新训练。
- severity=0 直接复用原数据路径，真实 checkpoint 输出差异为零。
- 所有统计沿用原 metrics 和标签方向，每个 severity 冻结该 checkpoint 的 validation 阈值。
- 已有 LEDGER clean 结果文件保持原样，所有新结果写入独立目录。
