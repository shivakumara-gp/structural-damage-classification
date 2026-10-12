# Structural Damage Image Classification — Local Results

## Reference comparison

The verified local results are compared with the paper and poster in [REFERENCE_COMPARISON.md](REFERENCE_COMPARISON.md), with a machine-readable table at [`results/reference_comparison.csv`](../results/reference_comparison.csv). Saved test predictions were checked against the local test labels. Keras and NumPy Tiny CNN training accuracy is available from saved histories; training accuracy was not saved for KNN, Logistic Regression, RBF SVM, or the layer-288 SVM. The reference scores are not directly comparable because the dataset and evaluation protocols differ. The report preserves the paper/poster MobileNetV1 discrepancy and the ambiguous reference split labels.

## Status

Dataset verification, three classical-model experiments, a small NumPy CNN, MobileNetV1, InceptionV3, Inception layer-288 SVM, and ResNet50 are complete on the local PEER Task 2 files. PyTorch and SciPy remain blocked by Windows Application Control, but TensorFlow 2.21.0 CPU was downloaded and loaded successfully over the new network. The pretrained Keras models ran on CPU; the RTX 3050 was not used.

The machine runs Python 3.13.12 on Windows 11 with 12 logical CPUs and an NVIDIA GeForce RTX 3050 GPU with 4 GB VRAM. The GPU is visible to the driver. Native Windows TensorFlow 2.21.0 reports no GPU support, so transfer experiments used CPU. PyTorch and SciPy imports remain blocked by Application Control; the existing NumPy implementations were used for the classical baselines and layer-288 SVM.

## Dataset and split

The verdict is **related PEER dataset, not the same 5,913-image dataset described in the Ho and Troncoso report**. The local README identifies this as PEER Hub ImageNet (Phi-Net), Task 2: Damage State; label 0 is Damaged and label 1 is Undamaged. The four `.npy` files passed exact header/byte-size, shape, dtype, finite-value, image-count, and one-hot label checks. Restoring the inferred BGR channel means to sampled rows gives integer-valued pixel values. The arrays are kept memory-mapped; train and test were never joined.

The split uses a seeded, stratified 10% validation subset drawn only from the 11,811 supplied training images (seed 42). The remaining 10,630 images fit KNN and logistic regression. The provided 1,460-image test set is reserved for final scoring. Class counts are 6,282/5,529 in training and 745/715 in test (Damaged/Undamaged). A SHA-256 scan found 0 exact duplicate groups and 0 exact cross-split duplicates among 13,271 image rows. This does not detect perceptual duplicates. The supplied README still does not prove image/label row ordering with an independent manifest.

## Preprocessing and implementation

Stored images were restored by adding `[103.939, 116.779, 123.68]` to the stored BGR channels, reversing to RGB, clipping to byte range, and resizing to 32×32 with Pillow bilinear resampling. Classical features then use the paper's fixed `x / 128 - 1` scaling. The mean-subtraction interpretation is supported by restored integer pixel samples but remains an inference. The reduced input resolution is a documented compute adaptation.

Windows Application Control blocked downloaded SciPy and PyTorch binary libraries, including when the import check ran outside the sandbox. Therefore, the three classical models use NumPy implementations. Logistic regression uses mini-batch Adam with L2 penalty `C=1`, learning rate 0.001, batch size 512, and validation-loss early stopping; it is not scikit-learn's LBFGS solver. RBF-SVM uses a NumPy SMO solver with `C=1`, `gamma=0.001`, and a reproducible 2,500-row subset of the 10,630 training rows because the full kernel is resource-intensive. KNN uses all 10,630 training rows and `k=5`.

The additional NumPy CNN uses two valid 3×3 convolutions with stride 2 (3→8 then 8→16 channels), ReLU activations, global average pooling, and a two-class linear output. It is trained from scratch with Adam (learning rate 0.003), batch size 32, for 12 epochs. It has no pretrained weights, augmentation, or fine-tuning.

The pretrained experiments use Keras ImageNet weights with frozen feature backbones and a trained two-class Dense head (SGD learning rate 0.01, up to 30 epochs). MobileNetV1 uses 224×224 inputs, InceptionV3 uses 299×299, and ResNet50 uses 224×224. Feature extraction uses batches of 8 and cached pooled embeddings. These CPU runs are transfer-learning linear probes, not end-to-end fine-tuning. The InceptionV3 model has 313 Keras layers; index 288 is `activation_90`, with an 8×8×384 activation. The layer-288 SVM uses spatially mean-pooled 384-dimensional activations, RBF `C=1`, `gamma=0.001`, and a seeded 2,500-row subset of the training split. Pooling and subset selection reduce memory and computation, so this is a documented adaptation of the paper's flattened-layer SVM.

Inference times below cover classifier prediction on cached features and exclude backbone feature extraction, image restoration, and resizing. Training times for pretrained models include feature extraction plus head fitting. MobileNetV1's 324.8 seconds includes its uncached feature extraction (321.6 seconds) and head fitting (3.2 seconds); a later cache-only rerun is much shorter. These elapsed times are local CPU measurements, not hardware-independent comparisons.

## Measured results

| Model | Validation accuracy | Test accuracy | Test precision (Damaged) | Test recall (Damaged) | Test F1 (Damaged) | Fit seconds | Inference ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN, k=5 | 59.10% | 55.27% | 55.31% | 64.30% | 59.47% | 0.08 | 0.65 |
| Logistic regression, L2 C=1 | 59.02% | 59.79% | 57.96% | 77.18% | 66.21% | 4.9 | 0.007 |
| RBF-SVM, C=1 gamma=0.001 | 65.20% | 61.30% | 59.47% | 75.84% | 66.67% | 0.59 | 0.067 |
| NumPy Tiny CNN (additional alternative) | 61.81% | 57.88% | 56.69% | 73.96% | 64.18% | 12.2 | 0.030 |
| MobileNetV1 (Keras ImageNet, frozen) | 84.17% | 82.40% | 79.26% | 88.72% | 83.72% | 324.8 | 0.048 |
| InceptionV3 (Keras ImageNet, frozen) | 82.64% | 81.16% | 78.94% | 86.04% | 82.34% | 1,055.5 | 0.057 |
| InceptionV3 layer-288 + RBF SVM | 71.30% | 69.32% | 65.92% | 82.55% | 73.30% | 1,052.2 | 0.034 |
| ResNet50 (Keras ImageNet, frozen; additional) | 84.17% | 83.77% | 81.51% | 88.19% | 84.72% | 1,034.1 | 0.060 |

Confusion matrices have actual classes as rows and predictions as columns, in `[Damaged, Undamaged]` order:

| Model | Confusion matrix |
|---|---|
| KNN | `[[479, 266], [387, 328]]` |
| Logistic regression | `[[575, 170], [417, 298]]` |
| RBF-SVM | `[[565, 180], [385, 330]]` |
| NumPy Tiny CNN | `[[551, 194], [421, 294]]` |
| MobileNetV1 | `[[661, 84], [173, 542]]` |
| InceptionV3 | `[[641, 104], [171, 544]]` |
| InceptionV3 layer-288 + SVM | `[[615, 130], [318, 397]]` |
| ResNet50 | `[[657, 88], [149, 566]]` |

Among the eight measured experiments, ResNet50 has the highest test accuracy (83.77%), followed by MobileNetV1 (82.40%) and InceptionV3 (81.16%). The layer-288 SVM scores 69.32% with the pooled-feature and subset adaptations stated above. These pretrained CNNs identify damaged and undamaged test examples more evenly than the classical baselines. The NumPy CNN remains a small model trained from scratch and is a separate comparison.

ResNet50 was included as a complementary ImageNet-pretrained CNN architecture, providing an additional comparison alongside the reference models MobileNetV1 and InceptionV3 under the same PEER Task 2 split and evaluation protocol. It achieved 83.77% test accuracy, 81.51% damaged-class precision, 88.19% damaged-class recall, and 84.72% damaged-class F1. These measurements describe this local PEER Task 2 experiment; the reference paper does not report a ResNet50 result, so they are not a direct comparison with a paper benchmark. The score alone does not establish that ResNet50 is superior because of its architecture.

Artifacts are under `artifacts/local/`: `all_model_results.csv` and JSON contain combined metrics; model-specific JSON files contain results and learning histories; PNGs contain the combined accuracy comparison and confusion matrices; `.npz` files store fitted NumPy models and Keras linear heads; feature caches contain the 32×32 classical inputs and pooled CNN embeddings.

## Reference comparison and limits

The report describes 5,913 labeled images and reports KNN 53.7%/52.9%, logistic regression 54.7%, and RBF-SVM 50.7%. The poster calls its listed scores validation accuracy and gives MobileNetV1 67.7%, while the report gives 67.0%. Test/validation attribution conflicts within the report. These are reference values only: this project uses a larger PEER Task 2 array set, a provided test partition, reduced-resolution classical features, and different classifier implementations. They are not directly comparable and do not constitute exact reproduction.

The requested four architectures now have local results using Keras and ImageNet weights. These runs are on PEER Task 2, not the paper's 5,913-image collection. Frozen backbones with linear heads differ from any paper fine-tuning, and the layer-288 SVM uses pooled features plus a subset. Inception layer numbering is library-version dependent; this run records Keras index 288 as `activation_90` in a 313-layer model. The PyTorch runner still cannot load its DLL, but PyTorch was not needed for these Keras CPU runs. Both optional notebooks have since been executed locally: the dataset-verification notebook ran checks, and the experiments notebook displayed saved results while all training switches remained disabled. Notebook execution did not generate the canonical model metrics; those come from the root runner scripts and saved result files.

## Reproduction

From PowerShell in the project root:

```powershell
python run_local.py --stage verify
python run_local.py --stage classical
python run_numpy_cnn.py
python run_keras_models.py --models mobilenet_v1 inception_pair resnet50
```

The Keras command downloads pretrained weights on first use and runs on CPU in this Windows configuration. It saves cached features, trained linear heads, predictions, metrics, and confusion matrices under `artifacts/local/`. Feature caches can be reused by later runs; cached reruns do not include the original feature-extraction time.
