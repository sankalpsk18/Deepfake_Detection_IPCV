# Engineering Prompt: Upgrade Deepfake Detection Prototype for Real-World Robustness

## ROLE
You are an ML engineer tasked with hardening an existing deepfake detection prototype so it generalizes beyond its benchmark dataset and holds up under real-world video/image conditions. You have full context on the current architecture below — do not redesign it from scratch, extend it.

## CURRENT SYSTEM (baseline — do not break this)
- **Primary detector**: EfficientNet-B0 (timm, ImageNet-pretrained), fine-tuned on FaceForensics++ (c23), 224×224 input, ImageNet normalization, 2-class output.
- **Secondary detector**: MesoNet-4, 256×256 input, [0,1] normalization, 2-class output.
- **Ensemble**: `final_probs = 0.7 × EfficientNet_probs + 0.3 × MesoNet_probs`, prediction via argmax.
- **Preprocessing**: MediaPipe FaceDetection (model_selection=1, conf=0.5) → 30% padded crop → aspect-preserving resize+pad.
- **Explainability**: Grad-CAM on EfficientNet `features.7`.
- **Runtime**: ONNX Runtime (CPU/CUDA), sequential inference.
- **Training data**: FaceForensics++ only (140k train / 39k val / 11k test), 4 manipulation types (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

## OBJECTIVE
Close the gap between benchmark accuracy and real-world accuracy by implementing three upgrades, in priority order:
1. Multi-dataset training with cross-generator generalization
2. Degradation-aware training/augmentation
3. Temporal consistency modeling (video-level signal, not just per-frame)

Each task below should be implementable independently and merged into the existing pipeline without breaking the ONNX export or Grad-CAM explainability path for the image-level branch.

---

## TASK 1 — Multi-Dataset Training & Cross-Generator Generalization

**Problem**: Training exclusively on FaceForensics++ means the model has only ever seen 4 older-generation (GAN/graphics-based) manipulation methods. It will fail on diffusion-based fakes and unseen face-swap tools it wasn't trained on, even if FF++ test accuracy is high.

**Requirements**:
- Add at least two additional datasets to the training pool:
  - **DFDC** (Deepfake Detection Challenge) — broader diversity of subjects, compression levels, and manipulation methods.
  - **Celeb-DF v2** — higher-quality/harder fakes than FF++, good stress test.
  - **One diffusion-based forgery dataset** (e.g. DF40, or another recent AIGC-face dataset) — critical, since diffusion-generated fakes are the dominant real-world threat and structurally different from GAN artifacts.
- Build a **unified data loader** that:
  - Normalizes labels across datasets to a common schema (`real` / `fake`, plus optional `manipulation_type` and `generator_family` metadata fields).
  - Applies **balanced sampling** across datasets and manipulation types per batch (avoid the model overfitting to whichever dataset is largest — FF++ dominates by volume otherwise).
  - Supports a config file (YAML/JSON) listing dataset paths, split ratios, and per-dataset sampling weights.
- Implement a **held-out generator-family validation protocol**: at least one manipulation method/generator family must be fully excluded from training and used only for validation, to explicitly measure generalization to unseen generators (not just random-split accuracy).
- Log and report **per-dataset and per-manipulation-type accuracy** separately, not just an aggregate number — aggregate accuracy hides which generator families are failing.

**Deliverables**: `dataset_config.yaml`, unified `MultiDatasetLoader` class, balanced `WeightedRandomSampler` or equivalent, generator-holdout validation split script.

---

## TASK 2 — Degradation-Aware Augmentation Pipeline

**Problem**: The model has only ever seen FF++'s native c23 compression. Real-world video/images arrive with arbitrary re-compression, resizing, and noise (e.g. after passing through social media platforms), and the model is likely brittle to this even when the underlying content is easy.

**Requirements**:
- Implement a stochastic augmentation pipeline (use `albumentations` or equivalent) applied **only during training**, with independently randomized per-sample application of:
  - **JPEG re-compression**: quality randomly sampled in range 30–95.
  - **Resolution downscale/upscale**: random downscale factor 0.25×–1.0×, then upscale back to model input size (simulates low-res source footage).
  - **Gaussian blur**: kernel size 3–7, sigma 0.1–2.0.
  - **Gaussian noise**: std sampled in range appropriate to pixel scale (e.g. 2–15 on 0–255 scale).
  - **Brightness/contrast/saturation jitter**: mild random jitter (±15–20%).
  - **Mild geometric augmentation**: small random rotation (±5°) and horizontal flip — keep conservative since face alignment matters for MesoNet's fixed-size input.
- Apply degradations with a **per-transform probability** (e.g. 30–50% chance each) rather than always stacking all of them, to preserve some clean-signal training examples too.
- Build a **robustness evaluation set**: take a held-out validation subset and generate multiple degraded copies at fixed severity levels (e.g. mild / moderate / severe), then report accuracy **as a function of degradation severity** — this curve is the actual metric that matters for production, not single-number clean accuracy.
- Keep MesoNet's `[0,1]` normalization and EfficientNet's ImageNet normalization pipelines separate but ensure both receive the *same* degraded image (apply degradation before branch-specific normalization).

**Deliverables**: `degradation_augmentations.py` (albumentations pipeline), `robustness_eval.py` producing an accuracy-vs-severity report/plot.

---

## TASK 3 — Temporal Consistency Modeling (video input only)

**Problem**: If inputs are video, per-frame classification with simple aggregation (e.g. majority vote or averaging) throws away one of the strongest deepfake signals: temporal inconsistency — unnatural blink timing, frame-to-frame flicker, boundary jitter around the face region across frames.

**Requirements**:
- Sample a short clip of **N consecutive frames** (start with N=16, sampled at a fixed stride, e.g. every 2nd frame) per video segment, using the existing MediaPipe face detection + crop pipeline per frame.
- Extract **per-frame embeddings** from EfficientNet-B0 (penultimate layer, before final classification head — do not use the raw class logits, use the feature vector).
- Feed the resulting embedding sequence into a lightweight temporal model — choose one:
  - **Option A (simpler)**: Bidirectional GRU/LSTM (1–2 layers, hidden size 128–256) with attention pooling over the sequence, followed by an FC classification head.
  - **Option B (stronger, more current)**: A small Temporal Transformer encoder (2–4 layers, 4–8 heads) over the frame embedding sequence with a learnable [CLS]-style pooling token.
- Optionally add **auxiliary temporal features** using landmarks already available from MediaPipe: eye-aspect-ratio sequence (for blink pattern anomalies) and mouth-landmark sequence (for lip-sync consistency), concatenated with the embedding sequence before the temporal model.
- **Fusion strategy**: combine the temporal model's clip-level prediction with the existing per-frame ensemble output using a **learned meta-classifier** (logistic regression or small MLP/gradient-boosted model trained on validation-set outputs) rather than a fixed weighted average — this generalizes better than hardcoded weights and lets the system learn when to trust temporal vs. per-frame signal.
- **Constraint**: keep this as a separate, optional branch in the pipeline (image-only inputs should still work through the existing per-frame ensemble path untouched). Note in your implementation that the temporal branch will complicate ONNX export (recurrent/transformer sequence models require careful export or a TorchScript fallback) — flag this as a known tradeoff rather than silently degrading the inference runtime.

**Deliverables**: `temporal_model.py` (GRU or Transformer option, configurable), frame-sequence sampler, updated `meta_classifier.py` for fusion, and a note on export path (ONNX vs TorchScript) for the temporal branch.

---

## ACCEPTANCE CRITERIA / EVALUATION PLAN
For the full upgraded system, report:
1. **In-domain accuracy** on FF++ test set (should not regress vs. baseline).
2. **Cross-dataset accuracy**: trained-on-FF++-only model evaluated on DFDC/Celeb-DF/diffusion set, vs. multi-dataset-trained model evaluated on the same — quantify the generalization gap closed.
3. **Held-out generator family accuracy** (Task 1 protocol) — this is the single most important number for claiming "generalization," not overall accuracy.
4. **Robustness curve**: accuracy vs. degradation severity (Task 2 output), clean vs. degraded comparison, before/after augmentation training.
5. **Video-level accuracy**: per-frame aggregation baseline vs. temporal-model-fused result on a video test set, ideally including at least one case designed to stress temporal artifacts (e.g. flicker-heavy fakes).
6. Confirm Grad-CAM explainability still functions on the EfficientNet branch after retraining, and confirm inference latency (CPU and GPU) for both the original per-image path and the new temporal path.

## CONSTRAINTS
- Do not remove or break the existing per-image ensemble path — treat all three tasks as additive upgrades with fallback to current behavior when video/temporal data isn't available.
- Keep code modular: dataset loading, augmentation, temporal modeling, and fusion should each be independently swappable/testable.
- Favor clarity and reproducibility (config files, seeded runs) over premature optimization.
