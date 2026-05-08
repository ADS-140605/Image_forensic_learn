# Graph Report - .  (2026-05-09)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 125 nodes · 129 edges · 27 communities (19 shown, 8 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f28c5e10`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Frontend Logic & State|Frontend Logic & State]]
- [[_COMMUNITY_Data Loading & Distribution|Data Loading & Distribution]]
- [[_COMMUNITY_Patch Extraction & Inference|Patch Extraction & Inference]]
- [[_COMMUNITY_UI Actions & Polling|UI Actions & Polling]]
- [[_COMMUNITY_CNN Architecture & Tests|CNN Architecture & Tests]]
- [[_COMMUNITY_Integral Image Utilities|Integral Image Utilities]]
- [[_COMMUNITY_Training Logic|Training Logic]]
- [[_COMMUNITY_Pipeline & Research Link|Pipeline & Research Link]]
- [[_COMMUNITY_Core AI Classes|Core AI Classes]]
- [[_COMMUNITY_Configuration Constants|Configuration Constants]]
- [[_COMMUNITY_UI Rendering|UI Rendering]]
- [[_COMMUNITY_Root Documentation|Root Documentation]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_Project Guidelines|Project Guidelines]]

## God Nodes (most connected - your core abstractions)
1. `PatchExtractor` - 8 edges
2. `CameraConvNet` - 8 edges
3. `IntegralImage` - 8 edges
4. `DresdenDataset` - 7 edges
5. `run_pipeline()` - 6 edges
6. `Trainer` - 6 edges
7. `CameraPredictor` - 5 edges
8. `startTraining()` - 5 edges
9. `runPrediction()` - 4 edges
10. `pollStatus()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `run_pipeline()` --calls--> `build_label_maps()`  [INFERRED]
  Homogeneous_Patches_CNN_v2/pipeline.py → Homogeneous_Patches_CNN_v2/AI/dataset/dresden.py
- `run_pipeline()` --calls--> `CameraConvNet`  [INFERRED]
  Homogeneous_Patches_CNN_v2/pipeline.py → Homogeneous_Patches_CNN_v2/AI/models/convnet.py
- `run_pipeline()` --calls--> `Trainer`  [INFERRED]
  Homogeneous_Patches_CNN_v2/pipeline.py → Homogeneous_Patches_CNN_v2/AI/training/trainer.py
- `DresdenDataset` --uses--> `PatchExtractor`  [INFERRED]
  Homogeneous_Patches_CNN_v2/AI/dataset/dresden.py → Homogeneous_Patches_CNN_v2/AI/dataset/patch_extractor.py
- `PatchExtractor` --uses--> `IntegralImage`  [INFERRED]
  Homogeneous_Patches_CNN_v2/AI/dataset/patch_extractor.py → Homogeneous_Patches_CNN_v2/AI/utils/integral_image.py

## Hyperedges (group relationships)
- **Hierarchical Classification Pipeline** — pipeline_py, trainer_py, convnet_py, distribution_py [EXTRACTED 1.00]
- **Patch Processing Subsystem** — patch_extractor_py, integral_image_py [EXTRACTED 1.00]

## Communities (27 total, 8 thin omitted)

### Community 0 - "Frontend Logic & State"
Cohesion: 0.07
Nodes (28): brandConfBar, brandConfPct, brandField, brandVoteChart, clearBtn, dropZone, fileInput, healthBadge (+20 more)

### Community 1 - "Data Loading & Distribution"
Cohesion: 0.15
Nodes (9): Dataset, build_label_maps(), DresdenDataset, dresden.py Dresden dataset wrapper., pipeline.py End-to-end training and evaluation pipeline., run_pipeline(), get_hierarchical_weights(), get_sampler() (+1 more)

### Community 2 - "Patch Extraction & Inference"
Cohesion: 0.22
Nodes (4): PatchExtractor, patch_extractor.py Logic for extracting and selecting homogeneous patches., CameraPredictor, predictor.py Inference helper for hierarchical classification.

### Community 3 - "UI Actions & Polling"
Cohesion: 0.28
Nodes (9): checkHealth(), pollStatus(), runPrediction(), setFile(), setLoading(), showToast(), startPolling(), startTraining() (+1 more)

### Community 4 - "CNN Architecture & Tests"
Cohesion: 0.29
Nodes (3): CameraConvNet, convnet.py Source Camera Identification ConvNet architecture from: "Camera mod, test_convnet_architecture()

### Community 8 - "Core AI Classes"
Cohesion: 0.29
Nodes (6): CameraConvNet, CameraPredictor, DresdenDataset, IntegralImage, PatchExtractor, Trainer

## Knowledge Gaps
- **44 isolated node(s):** `pipeline.py End-to-end training and evaluation pipeline.`, `config.py Configuration for the Camera Model Identification project.`, `dresden.py Dresden dataset wrapper.`, `patch_extractor.py Logic for extracting and selecting homogeneous patches.`, `predictor.py Inference helper for hierarchical classification.` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_pipeline()` connect `Data Loading & Distribution` to `CNN Architecture & Tests`, `Training Logic`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `PatchExtractor` connect `Patch Extraction & Inference` to `Data Loading & Distribution`, `Integral Image Utilities`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `DresdenDataset` connect `Data Loading & Distribution` to `Patch Extraction & Inference`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `PatchExtractor` (e.g. with `DresdenDataset` and `.__init__()`) actually correct?**
  _`PatchExtractor` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `CameraConvNet` (e.g. with `run_pipeline()` and `CameraPredictor`) actually correct?**
  _`CameraConvNet` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `IntegralImage` (e.g. with `PatchExtractor` and `.extract()`) actually correct?**
  _`IntegralImage` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `DresdenDataset` (e.g. with `run_pipeline()` and `PatchExtractor`) actually correct?**
  _`DresdenDataset` has 2 INFERRED edges - model-reasoned connections that need verification._