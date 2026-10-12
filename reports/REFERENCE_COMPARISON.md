# Comparison with the Ho and Troncoso Reference

## Scope and evidence

This comparison uses the project's saved local result files and the publicly available reference [paper](https://cs229.stanford.edu/proj2018/report/39.pdf) and [poster](https://cs229.stanford.edu/proj2018/poster/39.pdf).

All eight project test prediction arrays were compared directly with the supplied PEER Task 2 test labels (1,460 rows). Their calculated accuracies and confusion matrices match the saved results. The Keras histories save training accuracy at each epoch; the values below use the epoch with lowest validation loss (epoch 30 for the transfer models). The NumPy Tiny CNN training value uses its saved history at its best-validation-loss epoch (epoch 12). Classical-model and layer-288 SVM training accuracy summaries were not saved and are reported as **NOT SAVED**. The comparison CSV is `results/reference_comparison.csv`.

This is a benchmark comparison, not an exact reproduction. Our dataset is different, and our evaluation uses its own training/validation/test partitions. Because of these differences, the numeric accuracy gaps are **not treated as comparable percentage-point differences**.

## Consolidated results

| Model | Reference train accuracy | Reference evaluation accuracy | Local train accuracy | Local validation accuracy | Local test accuracy | Difference (pp) | Comparability and status |
|---|---:|---:|---:|---:|---:|---:|---|
| KNN, k=5 | 67.4% | 53.7% validation / 52.9% test (table split attribution is unclear) | NOT SAVED | 59.10% | 55.27% | N/C | Not directly comparable; local result verified. The paper says its main experiment did not reserve a test set, making the table's test attribution unclear. |
| Logistic Regression, L2 C=1 | 99.4% | 54.7% (main protocol implies validation) | NOT SAVED | 59.02% | 59.79% | N/C | Not directly comparable; local result verified. Different dataset, input resolution, split and solver. |
| RBF SVM, C=1, gamma=0.001 | 99.7% | 50.7% (main protocol implies validation) | NOT SAVED | 65.20% | 61.30% | N/C | Not directly comparable; local result verified. Local NumPy SMO uses a seeded 2,500-row fit subset. |
| NumPy Tiny CNN (additional) | Not reported | Not reported | 62.20%* | 61.81% | 57.88% | N/C | Additional local model; no reference result. *Training accuracy is the saved history value at epoch 12, the best validation-loss epoch. |
| MobileNetV1 | 80.0% | 67.0% paper; 67.7% poster (poster labels validation) | 86.08%* | 84.17% | 82.40% | N/C | Not directly comparable; local result verified. Reference discrepancy retained. Local frozen Keras ImageNet backbone plus linear head. |
| InceptionV3 | 81.0% | 83.0% (split attribution conflicts) | 84.43%* | 82.64% | 81.16% | N/C | Not directly comparable; local result verified. Paper abstract calls 83% test, poster calls it validation, while the paper's main protocol says no held-out test. |
| InceptionV3 layer-288 + RBF SVM | 95.0% | 75.0% validation (stated in paper prose and poster) | NOT SAVED | 71.30% | 69.32% | N/C | Not directly comparable; local result verified. Local layer 288 is Keras `activation_90` (8x8x384), spatially mean pooled to 384 features; fit uses 2,500 rows. |
| ResNet50 (additional) | Not reported | Not reported | 87.11%* | 84.17% | 83.77% | N/C | Additional local model; no reference result. |

*Saved training-history accuracy at the best-validation-loss epoch (epoch 30 for each pretrained Keras model). Classical and layer-288 SVM training accuracies are **NOT SAVED**.

**N/C** means no percentage-point difference is calculated because the reference and local measurements are not methodologically comparable. Local test accuracy is meaningful for comparisons among this project's models because they share the same held-out test partition; it should not be ranked directly against the reference paper's validation scores.

The paper's primary table gives the figures above. It also discusses separate Inception/Keras and augmentation experiments. Those are not substituted for the primary MobileNetV1 or InceptionV3 row values here. The paper separately describes an Inception diagnostic using 4,000 training images and 1,000 test images, with confusion matrix `[[448, 86], [117, 349]]`; this diagnostic is distinct from its stated main 5,913-image protocol.

## Dataset and protocol differences

- **Reference:** 5,913 labeled 224x224 8-bit RGB images; 3,186 undamaged and 2,727 damaged. The paper describes a stratified 90/10 training/validation split and states that it did not reserve a separate test set for the main experiment. The poster additionally lists 1,479 unlabeled images.
- **Local:** PEER Hub ImageNet (PHI-Net) Task 2 arrays: 11,811 training and 1,460 test images. Train labels are 6,282 damaged / 5,529 undamaged; test labels are 745 damaged / 715 undamaged. Mapping is `0 = Damaged`, `1 = Undamaged`; the paper describes the inverse numeric mapping (`0 = Undamaged`, `1 = Damaged`).
- **Local validation:** seeded, stratified 10% of the provided training data (1,181 rows); 10,630 rows remain in the fit partition. The provided 1,460-row test partition is kept separate in the runner code.
- **Image encoding:** local arrays are float32, 224x224x3. Adding the documented channel means yields integer-valued RGB image pixels after BGR reversal, consistent with BGR mean subtraction. This encoding interpretation is inferred from observed values.
- **Classical preprocessing:** local images are resized to 32x32 and scaled by `x/128 - 1`, rather than using full-resolution flattened images. Logistic regression is a NumPy mini-batch Adam implementation, not the paper's scikit-learn implementation. Classical RBF SVM uses a 2,500-row subset.
- **CNN preprocessing/training:** local MobileNetV1, InceptionV3 and ResNet50 use ImageNet weights with frozen feature backbones and a trained linear head, without augmentation. Local layer-288 SVM uses spatial mean pooling and a 2,500-row subset. These are documented compute/method adaptations.
- **Pairing and leakage:** local image and label counts match, and sampled pairs look plausible, but the package has no independent row-order manifest. Exact-byte duplicate scanning found no duplicate or train/test duplicate rows; perceptual duplicates were not checked. Thus exact duplication was not observed, but full sample correspondence and perceptual leakage cannot be ruled out.

These differences mean higher local values do not establish that a model outperforms the reference. Reference and local evaluation metrics are presented side by side for context only.

## Findings supported by the local evidence

1. **Best local result:** ResNet50 has the highest test accuracy among the eight local experiments at 83.77%. It is an extension rather than a reference-paper model. Among the six principal reference architectures/classifiers, MobileNetV1 has the highest local test accuracy at 82.40%.
2. **Transfer CNNs vs. classical models:** the three frozen ImageNet backbones score 81.16%–83.77% test accuracy, above the classical models' 55.27%–61.30% on this shared test partition. This is a supported within-project observation. The small NumPy CNN scores 57.88%, so the evidence does not support a blanket claim that every CNN beats every classical baseline.
3. **Why results differ:** dataset identity and sample count differ; the local test split is distinct from the paper's stated validation protocol; local class encoding is reversed but consistently mapped; classical resolution and solver differ; transfer backbones are frozen; and layer-288 pooling/subsampling differ. These are established methodological differences. Their individual causal effects have not been isolated experimentally.
4. **Other possible effects:** different random seeds, class balance, pretrained-weight versions, and augmentation could affect results. They are plausible factors, not causal findings from this experiment. The local runs used seed 42 and no augmentation.
5. **ResNet50:** it gives a useful additional same-dataset comparison because it used the same local split and evaluation pipeline. It cannot be compared to a paper score because the reference documents report no ResNet50 result.

## Reference table interpretation notes

- The paper's main protocol explicitly says there was no separate test set for the 5,913-image experiment. Its table header says “Validation/Test Accuracy”; the KNN row shows 53.7% / 52.9%. The report does not reconcile that row with the no-test statement. We preserve both values and flag attribution rather than treating 52.9% as a confirmed main-protocol test score.
- The paper table gives MobileNetV1 67.0%; the poster's table labels its evaluation column “Validation Accuracy” and gives 67.7%. Both are retained.
- The paper abstract describes InceptionV3's 83% as test accuracy; the poster table labels it validation accuracy. The main protocol says no separate test set, and the paper also describes a later 1,000-image Inception diagnostic. The 83% value's split is therefore marked ambiguous.
- The paper prose explicitly calls the layer-288 SVM's 75% value validation accuracy.
- The paper gives RBF SVM training accuracy as 99.7%; the poster table says 99.8%. The paper value is primary, and the poster discrepancy is recorded here.

## Reproducibility and artifact locations

The published summary values are in `artifacts/local/all_model_results.csv` and `.json`; model-specific result and history JSON files provide supporting details. Test accuracy and confusion matrices were checked against saved prediction arrays and local test labels during the audit. Prediction arrays, fitted snapshots, feature caches, and dataset arrays are local-only and are not part of the initial publication. Training-history values above are read from saved histories at the epoch with lowest validation loss. Classical and layer-288 SVM training accuracy summaries were not saved. Runner commands and configuration are documented in the root `README.md` and this report.

## Sources

- Reference paper: [Stanford CS229 paper PDF](https://cs229.stanford.edu/proj2018/report/39.pdf).
- Reference poster: [Stanford CS229 poster PDF](https://cs229.stanford.edu/proj2018/poster/39.pdf).
- Local dataset identity, array shape/count, class mapping, preprocessing caveats, duplicate scan, and split details: `reports/DATASET_VERIFICATION.md`, the locally generated dataset inventory and duplicate scan, and the experiment runners. The inventory files are local-only because they contain machine-specific paths.
