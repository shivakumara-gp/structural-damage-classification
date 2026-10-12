# Structural Damage Image Classification - Final Report

> Local experiments include the classical baselines, a small NumPy CNN, MobileNetV1, InceptionV3, Inception layer-288 + SVM, and ResNet50. Full methods and limitations are in [LOCAL_EXPERIMENT_REPORT.md](LOCAL_EXPERIMENT_REPORT.md); published values and split ambiguities are documented in [REFERENCE_COMPARISON.md](REFERENCE_COMPARISON.md).

## 1. Objectives

1. Reproduce the reference models as closely as possible, while documenting differences in dataset and implementation.
2. Evaluate additional models. A small NumPy CNN and ResNet50 were run as additional model comparisons.

## 2. Dataset and identity

We use PEER Hub ImageNet (Phi-Net) Task 2: Damage State. The local README maps 0 to Damaged and 1 to Undamaged. The data contain 11,811 training and 1,460 test samples at 224x224x3. This is not the paper's 5,913-image dataset, so this project is an alternative PEER Task 2 experiment rather than an exact reproduction. See `DATASET_VERIFICATION.md` for file integrity and remaining provenance limits.

## 3. Method

- Source split: keep PEER train and test separate.
- Validation: stratified 10% of training, random seed 42. Test is reserved for final evaluation.
- Stored image transform: values are consistent with BGR mean subtraction; restore to RGB then apply each architecture's preprocessing.
- Classical feature representation: resize RGB to 32x32, flatten, then apply the paper's fixed `x/128 - 1` scaling. The reduced resolution deviates from the paper's full-size flattened representation for resource reasons.
- Implemented additional model: small 32x32 NumPy CNN trained from scratch.
- Transfer models: ImageNet-pretrained Keras MobileNetV1, InceptionV3, and ResNet50 backbones were frozen; a linear two-class head was trained on pooled features.
- Hybrid: InceptionV3 layer-index 288 activations (`activation_90`, output 8×8×384) were spatially mean pooled and classified with an RBF SVM.
- Augmentation: none in the frozen-backbone baseline. Any future augmentation must affect training only.

## 4. Configurations and results

Full metrics, per-class reports, plots, and implementation caveats are in `LOCAL_EXPERIMENT_REPORT.md` and `artifacts/local/`.

| Model | Feature representation | Hyperparameters | Validation accuracy | Test accuracy | Test precision (damaged) | Test recall (damaged) | Test F1 (damaged) | Train time | Inference ms/image |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| KNN k=5 | 32x32 RGB, scaled | k=5 | 59.10% | 55.27% | 55.31% | 64.30% | 59.47% | 0.08 s | 0.65 |
| Logistic Regression | 32x32 RGB, scaled | L2, C=1; Adam, early stopping | 59.02% | 59.79% | 57.96% | 77.18% | 66.21% | 4.9 s | 0.007 |
| RBF SVM | 32x32 RGB, scaled | C=1, gamma=0.001; 2,500 fit rows | 65.20% | 61.30% | 59.47% | 75.84% | 66.67% | 0.59 s | 0.067 |
| NumPy Tiny CNN (additional alternative) | 32x32 RGB CNN features | Conv filters 8/16, Adam 0.003, 12 epochs | 61.81% | 57.88% | 56.69% | 73.96% | 64.18% | 12.2 s | 0.030 |
| MobileNetV1 | ImageNet frozen backbone, pooled features | Dense head, SGD 0.01, 30 epochs | 84.17% | 82.40% | 79.26% | 88.72% | 83.72% | 324.8 s | 0.048 |
| InceptionV3 | ImageNet frozen backbone, avg_pool features | Dense head, SGD 0.01, 30 epochs | 82.64% | 81.16% | 78.94% | 86.04% | 82.34% | 1,055.5 s | 0.057 |
| InceptionV3 layer 288 + SVM | `activation_90`, spatial mean pooled | RBF, C=1, gamma=0.001; 2,500 fit rows | 71.30% | 69.32% | 65.92% | 82.55% | 73.30% | 1,052.2 s | 0.034 |
| ResNet50 (additional) | ImageNet frozen backbone, pooled features | Dense head, SGD 0.01, 30 epochs | 84.17% | 83.77% | 81.51% | 88.19% | 84.72% | 1,034.1 s | 0.060 |

Training accuracy is available from saved histories for the Keras models and NumPy Tiny CNN. It is **NOT SAVED** for KNN, Logistic Regression, RBF SVM, and the InceptionV3 layer-288 SVM; no values are inferred for those models. Keras training used CPU TensorFlow. Classifier inference times use cached features and exclude backbone feature extraction.

## 5. Reference comparison

The report's Table 1 is headed "Validation/Test Accuracy" and gives KNN 53.7%/52.9%; the other table entries are single values: logistic regression 54.7%, RBF SVM 50.7%, MobileNetV1 67.0%, InceptionV3 83.0%, and layer-288 SVM 75.0%. The prose explicitly calls the layer-288 score validation accuracy. The report's main protocol says it held no test set, while the abstract calls Inception's 83% a test score and a later diagnostic section describes a separate 4,000-train/1,000-test confusion matrix. The poster labels its table as validation accuracy and reports MobileNetV1 at 67.7%. Preserve these conflicting descriptions rather than assigning an unsupported split to every figure. Sources: [paper](https://cs229.stanford.edu/proj2018/report/39.pdf), [poster](https://cs229.stanford.edu/proj2018/poster/39.pdf), [authors' code](https://github.com/jatron/structural-damage-recognition).

| Reference model | Reported train accuracy | Published evaluation value | Split attribution / note |
|---|---:|---:|---|
| KNN (k=5) | 67.4% | 53.7% / 52.9% | Table labels column Validation/Test; ambiguous against the report's no-test protocol |
| Logistic Regression | 99.4% | 54.7% | Single table value; split not separately marked |
| RBF SVM | 99.7% | 50.7% | Single table value; split not separately marked |
| MobileNetV1 | 80.0% | 67.0% report; 67.7% poster | Report table vs poster validation table |
| InceptionV3 | 81.0% | 83.0% | Abstract says test; main protocol says no held-out test; table does not resolve the conflict |
| InceptionV3 layer-288 + SVM | 95.0% | 75.0% | Prose explicitly says validation accuracy |

## 6. Discussion

See `LOCAL_EXPERIMENT_REPORT.md` for all measured models, preprocessing details, and implementation caveats. The transfer models ran with TensorFlow 2.21.0 on CPU; native Windows TensorFlow did not use the RTX 3050.

## 7. Limitations and conclusion

Confirmed limitations: different dataset size/release from the paper; dataset splits, preprocessing, model implementations, and evaluation protocols differ, so scores are not apples-to-apples; no independent sample-order manifest in the local README; classical inputs reduced to 32x32; custom NumPy solvers; RBF-SVM used a seeded subset; transfer backbones were frozen rather than fully fine-tuned; layer-288 activations were spatially pooled and its SVM used a seeded subset. Reference-paper evaluation split descriptions are ambiguous. ResNet50 is an additional model and has no corresponding reference-paper result. These are PEER Task 2 experiments, not an exact reproduction. See the local results report.

## 8. Reproduction

Run `python run_local.py --stage verify`, `python run_local.py --stage classical`, `python run_numpy_cnn.py`, and `python run_keras_models.py --models mobilenet_v1 inception_pair resnet50` from the workspace. The local runner stores results under `artifacts/local/`. Both optional notebooks have been executed locally; the experiments notebook displayed saved metrics with training switches disabled and did not generate the canonical experiment results.
