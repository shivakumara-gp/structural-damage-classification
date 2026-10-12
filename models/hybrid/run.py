"""Run the InceptionV3 and layer-288 SVM experiment pair.

The project runner computes the InceptionV3 pooled-head model and the layer-288
SVM together. Run from the project root: ``python -m models.hybrid.run``.
"""
import sys

from run_keras_models import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--models", "inception_pair"]
    main()
