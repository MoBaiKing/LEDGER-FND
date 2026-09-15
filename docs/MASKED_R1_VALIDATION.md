# R1 validation — actual execution, 2026-09-15

## Acceptance status

| Level | Status | Evidence / boundary |
|---|---|---|
| CPU engineering | PASS for executed checks | 66 pytest cases + 14 unittest subtests; actual tiny Qwen2/SigLIP/PEFT end-to-end; four-process Gloo; resume; CLI all; independent reload and diagnostics |
| GPU engineering | NOT RUN | User prohibited immediate GPU use. CUDA BF16/NCCL/full-backbone single/four-GPU smoke remain unexecuted. CPU BF16/SDPA is not a GPU substitute. |
| Real-data mechanism | NOT RUN / NOT SUPPORTED yet | Synthetic grouping/utility/relation diagnostics ran; no real OOF target quality, beneficial dissent, held-out utility, risk-coverage or robustness conclusion is supported. |
| Real-data performance | NOT RUN / NOT SUPPORTED yet | No new three-dataset metric or multi-seed improvement is claimed. Four-seed GPU suite is scheduled, not completed. |

## Executed tests and logs

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q --disable-warnings --tb=short`: **66 passed, 4 warnings, 14 subtests passed**, latest run 5.52 s. [Log](r1/cpu_tests.log).
- Initial R1 unit run: 26 passed, 3 failed. Two failures exposed a real BF16/FP32 relation-aggregation dtype mismatch. The ECE fixture used thresholds giving equal accuracy; it was changed to a threshold that actually changes decisions while ECE stays fixed. Failure log remains [preserved](r1/unit_first.log). No expected model property was weakened.
- Existing evaluation tests were updated only to patch the new `build_model` factory boundary instead of the old constructor name; their assertions remain intact.
- Actual tiny Qwen2 + SigLIP + PEFT, synthetic 36 train / 8 val / 8 test originals: reference, OOF, targets, student, average, threshold reselection, checkpoint reload. [Single CPU](r1/integration_cpu.log), [four-process Gloo](r1/integration_release_gloo4.log).
- Actual unified CLI with `--stage all --nproc 4`: [final CLI log](r1/final_cli_gloo4.log). It trained three reference folds, built targets, trained the student, and ran label-separated validation diagnostics. Default final test remained NOT RUN.
- Four-process Gloo `--resume .../last.pth --epochs 3` actually resumed a two-epoch synthetic run: [resume log](r1/resume_gloo4.log). This intentionally extends its budget; it is not an assertion of bit-identical metrics to a separately configured three-epoch schedule. RNG round-trip and optimizer-state tests cover the restoration components.
- Explicit synthetic-only `--stage test --frozen` executed: [log](r1/frozen_test_synthetic.log). This is not a real dataset test result.
- Actual validation mechanism CLI: [log](r1/diagnostics_cpu.log). Single-class tiny groups emit honest undefined-AUC warnings. No synthetic curve is presented as real evidence.
- Additional synthetic-only CLI checks ran the fixed-view exporter, pretrained judge, random judge and dataset-wide eval shuffle: [log](r1/frozen_controls_cpu.log). This checks executability, not pretrained value. The tiny exporter uses the last configured training epoch; no real controlled benchmark is claimed.
- Actual complete tiny Qwen/SigLIP encoder plus R1 forward/backward under CPU BF16 autocast: [log](r1/full_encoder_cpu_bf16.log), finite loss and gradients. CUDA AMP remains NOT RUN.
- The paired bootstrap CLI ran on one paired synthetic validation comparison with 20 resamples solely as an executable smoke: [output](r1/bootstrap_synthetic.json). This is not a reported significance result.
- `compileall`, shell/CLI help and `git diff --check` passed. No packages were installed or upgraded.

The host preloads a HAMI hook that prints initialization/cuInit=100 messages even when CUDA is hidden. All immediate model/training/test processes used `CUDA_VISIBLE_DEVICES=''`; actual training devices were CPU, and all recorded GPU allocations were zero. No GPU training was launched before the scheduled window.

## The 26 required contracts

Numbers correspond to the attachment. `tests/test_masked_r1_contract.py` contains test_01 … test_26 plus additional targeted cases.

| # | Actual CPU coverage | Result / missing part |
|---|---|---|
| 1 | target range/sum, Fake index NLL | PASS |
| 2 | same/opposite/unclear, wrong-but-agreeing example | PASS |
| 3 | asymmetric A/M exchange changes EDL | PASS |
| 4 | alpha/P/b/U identities, conflict belief | PASS |
| 5 | infeasible P_C=.9,U=.8 and uniform alpha=100 | PASS |
| 6 | KL vs torch.distributions, finite gradient, zero pairs | PASS |
| 7 | identical unsigned TV, opposite Delta signs | PASS |
| 8 | all 16 bit indices, fit-only prior, Shapley efficiency/symmetry/dummy | PASS |
| 9 | available-player subgame, probe mask after affine LN, no label signature | PASS |
| 10 | grouped outer/inner partitions disjoint, duplicates same fold | PASS; actual scheduled seeds additionally preflighted |
| 11 | deterministic augmentation, deliberate fingerprint mismatch rejection | PASS; actual three-dataset pixel/token replay checked |
| 12 | actual Qwen eager FP32, SDPA FP32/BF16 nonendpoint Jacobian | PASS on CPU; CUDA/BF16 NOT RUN |
| 13 | actual endpoint perturbation, invalid token states/bias isolation from G, two layers | PASS |
| 14 | batch sizes, single-view/no pairs/no G, explicit empty failure | PASS |
| 15 | logical-role/slot/mask/position-consistent permutation | PASS |
| 16 | CE blocked from utility, utility blocked from relation-specific params, L_rel active | PASS |
| 17 | multistep all-head + real PEFT all-layer LoRA gradient sources; all major ablations | PASS on tiny CPU |
| 18 | offline teacher no gradients, no old scorer/IURD names, shared parameter identity | PASS |
| 19 | simplex, monotonicity, zero missing weights | PASS |
| 20 | routing detach leaves forward values unchanged | PASS |
| 21 | remove/shuffle labels and IDs leaves predictions unchanged | PASS |
| 22 | student state-only reload, no teacher/cache dependency in inference | PASS |
| 23 | strict save/reload, actual Gloo four-process training/resume, CPU mixed BF16 autocast, ID dedup | PASS on CPU; GPU AMP/NCCL NOT RUN |
| 24 | label/threshold contract, ties, no test tuning, actual averaged model reselects | PASS |
| 25 | binary vs two-class Brier factor, top-label ECE independent of threshold decisions | PASS |
| 26 | positive-temperature ordering/AUC, calibration split guards and raw metadata rejection implementation | CPU unit coverage PASS; real val_cal calibration NOT RUN (default T=1) |

## Real data inspected on CPU

| Dataset | Train originals | Actual operations |
|---|---:|---|
| Weibo21 | 4,926 | manifest/group fold construction, backbone/tokenizer/image content hashes, real clean/augmentation pixel/token replay |
| Weibo | 5,415 | same |
| GossipCop | 16,069 | same |

Preflight logs: [Weibo21](r1/preflight_weibo21.log), [Weibo](r1/preflight_weibo.log), [GossipCop](r1/preflight_gossipcop.log). The earlier full-byte fingerprints record their audit-time source version; actual reference fitting recomputes the final version and seed-specific folds. Real preprocessing [results](r1/real_preprocessing_results.json), [log](r1/real_preprocessing.log). This did not perform a Qwen/SigLIP forward on real data.

## Unexecuted or limited items

- All real reference/student fitting, real inference, all real ablations, pretrained/random frozen-input comparison, held-out mechanism evaluation, perturbation evaluation and paired confidence intervals: **NOT RUN**.
- All actual GPU smoke/full training/performance measurements: **NOT RUN at delivery**. The scheduled queue may subsequently update its own live status.
- Full original legacy replay and matched legacy rerun: **NOT RUN**; legacy protocol regression tests passed, old code/config snapshot retained.
- Interrupted reference fitting preserves artifacts but has no mid-fold optimizer resume; use a fresh reference directory. Student resume is implemented and CPU-tested.
- Legacy typo corruption cannot guarantee entity or label preservation. No robustness improvement is claimed.
- Encoder-only controls have no trained relation branch; their uniform relation tensors are compatibility placeholders, not semantic estimates. Do not interpret their relationship diagnostics as a supervised relation model.
- Full-training student memory report is explicitly rank 0. Reference memory is the process peak observed by each fold's end. Resume resource statistics cover the current invocation; aggregate prior log segments for total resumed cost. No measured FLOPs claim exists.

## Paper claims

Supported: implementation of the stated target definitions/gradient policy and observed CPU numerical/locality/protocol properties within the tested APIs and synthetic shapes. The real-data preflight establishes input availability/replay, not model quality.

Unsupported: improved Macro-F1/calibration/robustness, reliable critical-minority recognition, independent human semantic accuracy, real-world causality, superiority of pretrained deliberation, five-seed significance, SOTA, or publication acceptance. Report future negative results and failed seeds without renaming metrics or selecting a favorable test result.
