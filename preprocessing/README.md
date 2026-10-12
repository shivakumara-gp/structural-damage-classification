# Preprocessing code locations

Preprocessing is currently implemented in the existing project runners rather than in duplicate modules here:

- `run_local.py`: restores the stored BGR mean-subtracted images, converts to RGB, resizes classical-model inputs to 32x32, and applies the classical feature scaling.
- `run_keras_models.py`: restores images and applies each Keras backbone's required input preprocessing while building feature batches and caches.
- `notebooks/02_experiments.ipynb`: optional notebook workflow with its own documented preprocessing cells.

The local image transform and its inferred BGR mean-subtraction convention are documented in `reports/LOCAL_EXPERIMENT_REPORT.md`. This folder is an index only; the runners remain the source of the executable preprocessing code.
