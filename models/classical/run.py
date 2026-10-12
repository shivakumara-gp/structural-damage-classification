"""Run the KNN, logistic-regression, and RBF-SVM baseline experiments.

This delegates to the canonical root runner so the model implementation and
artifact locations remain defined in one place.
Run from the project root: ``python -m models.classical.run``.
"""
from run_local import main as project_main


if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "--stage", "classical"]
    project_main()
