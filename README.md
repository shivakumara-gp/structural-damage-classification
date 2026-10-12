# structural-damage-classification

Local Windows/VS Code workflow for classifying structural images as **Damaged** or **Undamaged**. This project uses the supplied PEER Hub ImageNet (PHI-Net) Task 2 arrays. It does not use Google Colab.

The reference work is *Structural Damage Image Classification* by Minnie Ho and Jorge Troncoso ([paper PDF](https://cs229.stanford.edu/proj2018/report/39.pdf); [project poster PDF](https://cs229.stanford.edu/proj2018/poster/39.pdf)). Our available PEER Task 2 dataset differs from the paper's 5,913-image dataset, so these experiments are a comparison against the reference, not an exact reproduction.

## Dataset verdict

The local arrays are complete and match their NumPy headers, shapes, dtypes, and label counts. The registered classes are `0 = Damaged` and `1 = Undamaged`. They are a **related PEER Task 2 dataset, not the same 5,913-image collection described in the Ho and Troncoso paper**. The README supplied with the arrays does not independently prove image/label row order, and the precise Task 2 release is not established. No images are invented or concatenated; the supplied train and test partitions remain separate.

The dataset identity is **PEER Hub ImageNet (PHI-Net), Task 2: Damage State**. Place these four files in `dataset/`:

```text
dataset/task2_X_train.npy
dataset/task2_y_train.npy
dataset/task2_X_test.npy
dataset/task2_y_test.npy
```

The full evidence and remaining provenance caveats are in [`reports/DATASET_VERIFICATION.md`](reports/DATASET_VERIFICATION.md). Measured model results are in [`reports/LOCAL_EXPERIMENT_REPORT.md`](reports/LOCAL_EXPERIMENT_REPORT.md). The paper comparison is in [`reports/REFERENCE_COMPARISON.md`](reports/REFERENCE_COMPARISON.md), with its machine-readable table at [`results/reference_comparison.csv`](results/reference_comparison.csv).

## Project files

```text
dataset/                 Supplied arrays, README, and license (large arrays stay out of Git)
src/                     Local prediction utility
run_local.py             Dataset audit and classical NumPy baselines
run_numpy_cnn.py         Additional small CNN trained using NumPy only
run_keras_models.py      ImageNet Keras backbones and Inception layer-288 SVM
run_local_deep.py        Deferred PyTorch transfer-learning runner
check_dataset.py         Convenient entry point for the audit
models/                  Model runner entry points and local binary snapshots
preprocessing/           Index to preprocessing code in the existing runners
artifacts/local/         Published metric summaries/plots; arrays, predictions, caches, and weights stay local
reports/                 Dataset audit, experiment report, final report, comparison
results/                 Curated reference comparison CSV
notebooks/               Optional local notebooks; excluded from GitHub
```

The source runners stay at the project root for compatibility with the documented commands. Convenience entry points are also available under `models/`; see [`models/README.md`](models/README.md). Preprocessing is currently implemented in the runners and indexed in [`preprocessing/README.md`](preprocessing/README.md); there is no separate tuning package because no systematic hyperparameter search was performed.

## Open and set up in VS Code

1. In VS Code, open the project folder you cloned or extracted.
2. Open **Terminal → New Terminal** and select PowerShell.
3. Create a virtual environment and install the core requirements:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select `.venv\Scripts\python.exe`. The workspace setting also points VS Code at that interpreter.

If PowerShell activation is permitted, you can activate it with `\.venv\Scripts\Activate.ps1`. Otherwise, call `.\.venv\Scripts\python.exe` directly as in the commands below.

The core local requirements are NumPy, Pillow, and Matplotlib. Install `requirements-keras.txt` for the pretrained CNN runs. TensorFlow 2.21.0 runs on CPU in this native Windows setup; model weights download on first use. The separate PyTorch runner remains blocked by Windows Application Control, but it is not needed for the measured Keras experiments.

## Run the local workflow

Run these commands from the project root in PowerShell. If the virtual environment is selected in VS Code, `python` uses it; otherwise use `.\.venv\Scripts\python.exe` in place of `python`.

1. Audit the arrays and scan for exact duplicate rows:

   ```powershell
   python check_dataset.py
   ```

   The check uses memory mapping, validates file sizes/header metadata, shapes, dtypes, finite values, and one-hot labels, then writes the audit and duplicate scan to `artifacts/local/`.

2. Run KNN, logistic regression, and RBF-SVM baselines:

   ```powershell
   python run_local.py --stage classical
   ```

   The training data are split 90/10 with stratification and seed 42. The provided test partition is held out. Classical features are RGB resized to 32×32 and scaled with `x/128 - 1`. The RBF-SVM uses a reproducible 2,500-row fit subset to keep its kernel calculation manageable; this deviation is recorded in the metrics.

3. Train the additional NumPy-only CNN:

   ```powershell
   python run_numpy_cnn.py
   ```

   It reads the generated feature cache, trains a small two-convolution CNN from scratch, and saves its learning curve, predictions, metrics, and weights under `artifacts/local/`. It is an alternative model, not a ResNet50 or paper architecture.

4. Run MobileNetV1, InceptionV3, Inception layer-288 + SVM, and ResNet50:

   ```powershell
   python -m pip install -r requirements-keras.txt
   python run_keras_models.py --models mobilenet_v1 inception_pair resnet50
   ```

   These use ImageNet-pretrained frozen Keras backbones with a trained linear head. The layer-288 SVM uses spatially pooled `activation_90` features and a seeded 2,500-row subset. Feature caches, predictions, metrics, and confusion matrices are saved under `artifacts/local/`.

5. Load a saved NumPy model and classify an RGB image:

   ```powershell
   python src/predict_local.py "path/to/building.jpg"
   python src/predict_local.py "path/to/building.jpg" --model cnn
   ```

   `--model` choices are `all`, `knn`, `logistic`, `svm`, and `cnn`. The command prints the predicted class. Saved model parameters are loaded from `artifacts/local/`.

## Results and interpretation

The local validation and test accuracy results are:

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| KNN (k=5) | 59.10% | 55.27% |
| Logistic Regression | 59.02% | 59.79% |
| RBF SVM | 65.20% | 61.30% |
| NumPy Tiny CNN (additional) | 61.81% | 57.88% |
| MobileNetV1 | 84.17% | 82.40% |
| InceptionV3 | 82.64% | 81.16% |
| InceptionV3 layer-288 + SVM | 71.30% | 69.32% |
| ResNet50 (additional) | 84.17% | 83.77% |

Full metrics and methods are in [`reports/LOCAL_EXPERIMENT_REPORT.md`](reports/LOCAL_EXPERIMENT_REPORT.md). The latest local comparison is in `artifacts/local/all_model_results.csv`; the chart is `artifacts/local/all_model_accuracy_comparison.png`. The directory `artifacts/local/` also contains per-model result JSON, saved test predictions, confusion matrices, fitted NumPy models and Keras linear heads, feature caches, and training histories where applicable. Project reports are in `reports/`; the reference comparison CSV is in [`results/reference_comparison.csv`](results/reference_comparison.csv). The Keras head files do not include complete standalone backbones; Keras ImageNet weights are downloaded when needed. Local training accuracy was **NOT SAVED** for KNN, Logistic Regression, RBF SVM, or the InceptionV3 layer-288 SVM. TensorFlow experiments ran on CPU; the listed Keras inference timings use cached features and exclude backbone feature extraction. Results from this PEER Task 2 data must not be presented as an exact reproduction of the paper.

The reference comparison is maintained as a report and CSV; there is no comparison-generation script. To update it, use the saved metrics in `artifacts/local/all_model_results.csv` and `artifacts/local/all_model_results.json`, check test metrics against the saved predictions and `dataset/task2_y_test.npy`, then update `reports/REFERENCE_COMPARISON.md` and `results/reference_comparison.csv` together. Keep reference validation and local test results distinct, and mark differences `N/C` when the protocols are not comparable.

The current comparison includes all four requested architectures. The PyTorch implementation remains unavailable under the current Windows policy; the successful runs use TensorFlow/Keras CPU instead. Classifier inference times measure the prediction head on cached features and exclude CNN feature extraction.

## Version control

The `.gitignore` selectively permits small result CSV/JSON summaries and metric plots while excluding dataset arrays and photos, prediction arrays, feature caches, model snapshots, downloaded wheels, and virtual environments. Keep `dataset/license.txt` with the dataset metadata; it does not license this project's source code. A fresh clone must provide the four dataset arrays locally before running the experiments; full results and checkpoints can be recreated by the documented runners. The root runners remain the canonical implementations, and the `models/` entry points are convenience launchers..
