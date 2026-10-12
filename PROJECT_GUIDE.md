# Structural Damage Image Classification

## Dataset decision

The local `dataset/README.txt` identifies the arrays as PEER Hub ImageNet (Phi-Net), Task 2: Damage State. It maps label index 0 to Damaged and index 1 to Undamaged. These files contain 11,811 training rows and 1,460 test rows. Each label file has the matching number of rows.

This is not the reference paper's 5,913-image collection. Results must be described as experiments on PEER Task 2, not as exact reproduction. Do not combine the provided train and test partitions. A stratified 10% validation set is taken only from the provided training partition.

The arrays are complete by NumPy header and byte-size checks. The image data are finite 224x224x3 float32 tensors. Restoring `[103.939, 116.779, 123.68]` to sampled channels produces exact integer pixel values; this is consistent with Caffe-style BGR mean subtraction. The notebook reverses this transform and channel order before applying model-specific preprocessing. This storage interpretation is inferred from the pixel values.

The supplied README confirms the label mapping but does not document row ordering or identify a precise Task 2 release version. File names, matching counts, and sample previews support pairing, but are not an independent proof of the original registration order. This residual limitation is recorded in the report.

## Run on this Windows PC

The supplied arrays are already in `dataset/`; no Colab or Drive upload is needed. Run commands from the project folder in PowerShell.

The NumPy-only local scripts need the packages in `requirements-local.txt`. The pretrained Keras experiments use `requirements-keras.txt`. PyTorch-based code has separate optional dependencies in `requirements-deep.txt`, but is not required.

Install the local requirements if needed:

```powershell
python -m pip install -r requirements-local.txt
```

1. Recheck files without starting a model:

   ```powershell
   python run_local.py --stage verify
   ```

2. Run KNN, logistic regression, and RBF-SVM baselines:

   ```powershell
   python run_local.py --stage classical
   ```

   The training images are read through a memory map. Small 32x32 feature arrays and reports are saved under `artifacts/local/`. The SVM uses a reproducible 2,500-image training subset because its full kernel is expensive; this is recorded in the output. The classical models use NumPy code because Windows Application Control blocks the compiled SciPy and PyTorch libraries on this machine.

3. Run the pretrained Keras CNNs and Inception layer-288 SVM:

   ```powershell
   python -m pip install -r requirements-keras.txt
   python run_keras_models.py --models mobilenet_v1 inception_pair resnet50
   ```

   TensorFlow runs on CPU in this native Windows configuration. ImageNet weights are downloaded on first use. Backbones are frozen and a linear head is trained; the layer-288 SVM uses spatially pooled features and a seeded 2,500-row fit subset.

4. Run the NumPy-only CNN fallback and combined model comparison:

   ```powershell
   python run_numpy_cnn.py
   ```

   This trains a small additional CNN from scratch on the cached 32x32 images. It is not a replacement for the paper's architectures or pretrained ResNet50.

The notebook versions remain available as optional Colab workflows. Large source arrays and generated artifacts are excluded from Git.

## Local data inventory

| File | Shape | Dtype | Bytes | Status |
|---|---|---|---:|---|
| `task2_X_train.npy` | `(11811, 224, 224, 3)` | `float32` | 7,111,544,960 | Complete by header and size |
| `task2_y_train.npy` | `(11811, 2)` | `float32` | 94,616 | One-hot; 6,282 damaged / 5,529 undamaged |
| `task2_X_test.npy` | `(1460, 224, 224, 3)` | `float32` | 879,083,648 | Complete by header and size |
| `task2_y_test.npy` | `(1460, 2)` | `float32` | 11,808 | One-hot; 745 damaged / 715 undamaged |
| `README.txt` | - | text | 203 | Dataset/task and label registration |
| `dataset/license.txt` | - | text | 20,849 | Present; retain with project |

## Reference comparison

The paper describes a 90/10 training/validation split on 5,913 labeled images and says no separate test set was held out. Its abstract nevertheless calls the 83% InceptionV3 result a test accuracy. The paper and poster tables mostly label their reported model scores as validation accuracy; the report lists MobileNetV1 at 67.0% while the poster lists 67.7%. These figures are references only and are not results from this project.

The authors' repository includes classical classifiers and multiple TensorFlow/Keras transfer-learning variants, rather than one fully specified final configuration for every architecture. The experiment notebook records its own deviations (32x32 classical input, frozen ImageNet backbones, PEER Task 2 split) and never claims exact reproduction.

## Project stages

- Stage 1: dataset inventory, file integrity, labels and sample preview - prepared in notebook 01.
- Stage 2: preprocessing, deterministic validation split and training-ready code - prepared in notebook 02.
- Stage 3: classical baselines and a small NumPy CNN alternative - run locally; results are in `reports/LOCAL_EXPERIMENT_REPORT.md`.
- Stage 4: MobileNetV1, InceptionV3, Inception layer-288 SVM, and ResNet50 - measured locally with TensorFlow/Keras CPU; results are in the local experiment report.
- Stage 5: local results and reference comparison - see `reports/LOCAL_EXPERIMENT_REPORT.md`, `reports/REFERENCE_COMPARISON.md`, and `reports/FINAL_REPORT_TEMPLATE.md`.
