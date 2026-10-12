"""Run one or more existing Keras transfer-learning experiments.

Pass the runner's accepted model names after this script, for example:
``python -m models.transfer_heads.run --models mobilenet_v1 resnet50``.
The canonical implementation and saved artifacts remain in the root runner.
"""
import sys

from run_keras_models import main


if __name__ == "__main__":
    main()
