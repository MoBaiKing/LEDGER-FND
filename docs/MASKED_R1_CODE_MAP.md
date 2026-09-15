# Formula and implementation map

| Definition | Actual implementation |
|---|---|
| A/M/C targets | [mmfnd/relation_supervision.py:8](/data/dyl/sotamodelv3_mask/mmfnd/relation_supervision.py:8) |
| Dirichlet P/U/b and conflict belief | [mmfnd/relation_supervision.py:14](/data/dyl/sotamodelv3_mask/mmfnd/relation_supervision.py:14) |
| Dirichlet KL | [mmfnd/relation_supervision.py:38](/data/dyl/sotamodelv3_mask/mmfnd/relation_supervision.py:38) |
| soft weighted EDL | [mmfnd/relation_supervision.py:46](/data/dyl/sotamodelv3_mask/mmfnd/relation_supervision.py:46) |
| 16 subsets / signed LOO / exact Shapley | [mmfnd/evidence_utility.py:12](/data/dyl/sotamodelv3_mask/mmfnd/evidence_utility.py:12) |
| OOF RMS scale | [mmfnd/evidence_utility.py:42](/data/dyl/sotamodelv3_mask/mmfnd/evidence_utility.py:42) |
| masked simplex weights | [mmfnd/evidence_utility.py:51](/data/dyl/sotamodelv3_mask/mmfnd/evidence_utility.py:51) |
| utility head | [mmfnd/evidence_utility.py:59](/data/dyl/sotamodelv3_mask/mmfnd/evidence_utility.py:59) |
| strict availability mask | [mmfnd/revision_masked_r1.py:17](/data/dyl/sotamodelv3_mask/mmfnd/revision_masked_r1.py:17) |
| per-layer shared Qwen mask | [mmfnd/revision_masked_r1.py:68](/data/dyl/sotamodelv3_mask/mmfnd/revision_masked_r1.py:68) |
| single fusion + CE stop-gradient | [mmfnd/revision_masked_r1.py:126](/data/dyl/sotamodelv3_mask/mmfnd/revision_masked_r1.py:126) |
| total loss | [mmfnd/losses_masked_r1.py:7](/data/dyl/sotamodelv3_mask/mmfnd/losses_masked_r1.py:7) |
| reference subset CE | [mmfnd/reference_subset_probe.py:25](/data/dyl/sotamodelv3_mask/mmfnd/reference_subset_probe.py:25) |
| fit-only empty prior | [mmfnd/reference_subset_probe.py:35](/data/dyl/sotamodelv3_mask/mmfnd/reference_subset_probe.py:35) |
| encoder-level reference cross-fit | [mmfnd/reference_pipeline.py:95](/data/dyl/sotamodelv3_mask/mmfnd/reference_pipeline.py:95) |
| label-free reference prediction | [mmfnd/reference_pipeline.py:63](/data/dyl/sotamodelv3_mask/mmfnd/reference_pipeline.py:63) |
| separate target building | [mmfnd/reference_pipeline.py:203](/data/dyl/sotamodelv3_mask/mmfnd/reference_pipeline.py:203) |
| strict cache matching | [mmfnd/r1_cache.py:126](/data/dyl/sotamodelv3_mask/mmfnd/r1_cache.py:126) |
| threshold and frozen model identity | [mmfnd/evaluation_masked_r1.py:58](/data/dyl/sotamodelv3_mask/mmfnd/evaluation_masked_r1.py:58) |
