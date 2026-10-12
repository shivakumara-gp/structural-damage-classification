# Saved model files

This directory contains runner entry points and local copies of some saved model files from `artifacts/local/`. The `.npz` snapshots are local-only generated binary artifacts and are excluded by `.gitignore`; the initial public repository does not include these snapshots. Some model snapshots may be absent here or under `artifacts/local/`. The Python entry points are source files and can be committed.

## Contents

- `classical/`: fitted NumPy KNN, Logistic Regression, and RBF SVM parameters; `run.py` launches the existing classical runner.
- `numpy_cnn/`: trained NumPy Tiny CNN parameters; `run.py` launches its existing runner.
- `transfer_heads/`: trained linear classification heads for MobileNetV1, InceptionV3, and ResNet50; `run.py` forwards accepted model choices to the existing Keras runner.
- `hybrid/`: fitted RBF SVM for spatially pooled InceptionV3 layer-288 features; `run.py` launches the existing Inception model pair.

Run commands from the project root:

```powershell
python -m models.classical.run
python -m models.numpy_cnn.run
python -m models.transfer_heads.run --models mobilenet_v1 resnet50
python -m models.hybrid.run
```

These are convenience entry points, not duplicate implementations. They invoke the root scripts, which own the model code, train the selected experiments, and generate experiment outputs and model snapshots under `artifacts/local/`. They do not populate this directory with snapshot copies. Running them can take substantial time and may download pretrained ImageNet weights for the Keras backbones.

The three files in `transfer_heads/` contain classifier-head parameters, not complete standalone networks. Their corresponding frozen ImageNet backbones are defined by the project code and require the Keras ImageNet weights. To reproduce predictions with the existing workflow, use the project runners and their expected files under `artifacts/local/`.

The runner entry points expect data under `dataset/` and generate files under `artifacts/local/`, as documented in the root README. Re-running the relevant root script generates fresh outputs there; it does not create the duplicate snapshot copies shown in this local directory. Do not assume any model snapshot is included in a clone. Model snapshots do not include training accuracy for KNN, Logistic Regression, RBF SVM, or the layer-288 SVM. Refer to the reports for methods, metrics, and limitations.
