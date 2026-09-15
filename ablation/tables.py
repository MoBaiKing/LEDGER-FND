"""Generate reviewable design tables and unfilled result templates from registry."""
import csv
import argparse
import json
from pathlib import Path

from ablation.registry import ROOT, VARIANTS, CONTROLS, LEGACY, states, settings, unavailable_losses

LABELS = {
    "full": "原始完整模型", "mlp_classifier": "整体MLP分类器", "non_llm_evaluator": "随机因果Transformer",
    "no_uncertainty": "去显式关系不确定性", "no_critical_minority": "去Critical Minority",
    "no_global_token": "G改为有效pair均值", "fixed_mixture": "固定混合λ=0.5",
    "direct_only": "严格Direct", "deliberation_only": "仅Deliberation", "no_decision_correction": "去最终校正",
}


def generate():
    directory = ROOT / "ablation"
    from ablation.run import training_source
    seed_counts = "、".join(f"{dataset}={len(training_source(dataset)[1])}" for dataset in ("weibo21", "weibo", "gossipcop"))
    with (directory / "module_states.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        rows = [states(name) for name in VARIANTS + CONTROLS]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# 完整消融 Tables", "", "此表由 `python -m ablation.tables` 从当前 registry 生成。性能值仅来自实际训练，未运行不填数值。", "",
             "## Table 1：Full基线与九个消融实验的真实模块设置", "",
             "所有行保留原始 T/V/E/X 编码。D 表示 Direct 候选，L 表示 Deliberation 候选；U 为显式关系不确定性。", "",
             "| 编号 | Variant | Evaluator | D | L | U | Minority | Global摘要 | 混合/整体分类 | 原始均值旁路 | 最终校正 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    yes = lambda value: "✓" if value else "—"
    aliases = {value: key for key, value in LEGACY.items()}
    evaluator = {"none": "不执行", "shared_qwen": "共享Qwen", "random_transformer": "随机Transformer"}
    global_name = {"none": "—", "learned_token": "Global token", "masked_pair_mean": "有效pair均值"}
    mix = {"not_applicable": "整体MLP分类", "direct": "Direct", "deliberation": "Deliberation", "fixed": "λ=.5", "dynamic": "动态λ"}
    for name in VARIANTS:
        s = states(name)
        row = [aliases.get(name, "基线"), name, evaluator[s["evaluator"]], yes(s["direct_candidate"]), yes(s["deliberation_candidate"]),
               yes(s["explicit_relation_uncertainty"]), yes(s["critical_minority"]), global_name[s["global_summary"]],
               mix[s["mixture"]], yes(s["raw_mean_residual"]), yes(s["decision_correction"])]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "λ为Deliberation系数：`(1−λ)D+λL`。strict Direct 的 confidence 槽位置0；它与原Full中的Direct不完全相同。", "",
              "## Table 2：依赖变化与损失", "", "| Variant | 连带变化 | 因依赖移除而关闭的loss |", "|---|---|---|"]
    notes = {
        "full": "完全沿用原实现", "mlp_classifier": "concat([T,V,E,X])→LN→Linear→GELU→Dropout→Linear(2)，整体替换原分类模块",
        "non_llm_evaluator": "结构、容量和预训练条件同时变化；Qwen仍用于文本编码",
        "no_uncertainty": "关系alpha/p及监督保留；u源头置0，乘法(1−u)变1",
        "no_critical_minority": "不再执行minority head，adjudication重新对有效证据softmax",
        "no_global_token": "G实际删除，pair槽位4..9不变；保留原global输出head接收masked mean",
        "fixed_mixture": "只替换最终混合系数；内部注意力、U和校正不变",
        "direct_only": "不执行evaluator；latent confidence固定0，raw-mean旁路及IURD关闭",
        "deliberation_only": "不执行Direct scorer；raw-mean最终旁路关闭；IURD只使用其原有输入",
        "no_decision_correction": "直接preliminary logits；保留final_norm/classifier，跳过calibrator内部LN/残差/温度",
    }
    for name in VARIANTS:
        lines.append(f"| {name} | {notes[name]} | {', '.join(unavailable_losses(name)) or '无'} |")
    lines += ["", "当前三个数据集classification=1，辅助loss均为0。表中关闭项即使将来base配置启用，也会显式禁用并记录原因。", "",
              "## Table 3：公平路径对照组", "", "| 配置 | Direct | Deliberation | Direct confidence | 均值旁路 | 校正 | 是否独立训练 |",
              "|---|---|---|---|---|---|---|",
              "| path_direct_reference | ✓ | — | 固定0 | — | — | 与direct_only等价，复用 |",
              "| path_deliberation_reference | — | ✓ | 不适用 | — | — | 是 |",
              "| dual_path_reference | ✓ | ✓ | 固定0 | — | — | 是；不等于Full |", "",
              "三者共用fusion_norm→final_norm→classifier。主表下游不完全一致，不把Full优势单独写成纯路径互补证据。", "",
              "## Table 4：正式性能表（当前尚未运行）", "",
              f"当前训练源seed数量：{seed_counts}；各数据集按实际数量生成任务。`—`表示未测量，不表示0。", "",
              "| Variant | Weibo21 Macro-F1 | Weibo Macro-F1 | GossipCop Macro-F1 | 实测状态 |", "|---|---|---|---|---|"]
    for name in VARIANTS:
        lines.append(f"| {name} | — | — | — | 本消融系统尚未正式训练 |")
    lines += ["", "正式运行后，`ablation.summarize` 会生成每数据集的完整Macro-F1/Accuracy/Fake与Real P/R/F1/AUC/NLL/Brier/ECE-15表，",
              "并附逐seed、mean±sample std、Full同seed差值（原始小数及百分点）、困难子集样本数/类别分布、成本和失败清单。", "",
              "## Table 5：成本记录口径", "", "| 项目 | 来源 / 当前状态 |", "|---|---|",
              "| 总参数、可训练参数 | 每次实际构造模型后写入config，当前未加载完整7B模型测量 |",
              "| LG-LED/替代Transformer/适配层参数 | 模块构造后独立计数，见head_parameter_counts.csv |",
              "| evaluator token数 | Full 11；no_global_token 10；整体MLP分类/direct 0 |",
              "| peak CUDA memory、wall time | 正式run的runtime_rank*.json；CPU测试不能代替7B成本结论 |",
              "| FLOPs | 未测量；不能从参数数直接推断 |", "",
              "没有任何表格预设Full最佳或预设某项下降幅度。解释边界及实际文件/函数映射见AUDIT.md。"]
    (directory / "TABLES.md").write_text("\n".join(lines) + "\n")


def parameter_counts():
    from ablation.model import AblationDeliberation, MLPClassificationModule
    from ablation.registry import resolve_config
    from mmfnd.latent_evidence_deliberation import LLMGuidedLatentEvidenceDeliberation
    from mmfnd.model import UncertaintyCalibratedDecision
    base = json.loads((ROOT / "configs/datasets/weibo21.json").read_text())
    dim = base["model"]["hidden_dim"]
    hidden = json.loads((ROOT / base["model"]["text_backbone"] / "config.json").read_text())["hidden_size"]
    rows = []
    count = lambda module: sum(p.numel() for p in module.parameters())
    for name in VARIANTS:
        cfg = resolve_config(base, name, base["seed"], "unused")
        if name == "mlp_classifier":
            module = None
            decision = MLPClassificationModule(
                dim, cfg["model"]["dropout"], cfg["model"].get("view_dropout_probability", 0.0)
            )
        else:
            module = LLMGuidedLatentEvidenceDeliberation(dim, hidden, cfg["model"]["lgled"])
            module.__class__ = AblationDeliberation
            module.configure(cfg)
            decision = UncertaintyCalibratedDecision(dim)
            if not module.state["decision_correction"]:
                decision.calibrator = None
        rows.append({"variant": name, "evidence_dim": dim, "qwen_hidden_dim": hidden,
            "lgled_parameters": count(module) if module is not None else 0, "decision_parameters": count(decision),
            "replacement_transformer_parameters": count(module.evaluator.layers) + count(module.evaluator.norm) if module is not None and hasattr(module, "evaluator") else 0,
            "replacement_adapter_parameters": count(module.evaluator.input_adapter) + count(module.evaluator.output_adapter) if module is not None and hasattr(module, "evaluator") else 0,
            "evaluator_tokens": 0 if module is None or module.state["evaluator"] == "none" else (10 if name == "no_global_token" else 11),
            "scope": "head modules only; excludes shared Qwen/SigLIP/upstream encoder; no FLOPs or performance claim"})
    with (ROOT / "ablation/head_parameter_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-counts", action="store_true", help="Also instantiate CPU heads for exact parameter counts; no backbone loading")
    args = parser.parse_args()
    if args.parameter_counts:
        parameter_counts()
    generate()
