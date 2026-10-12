# Dataset Verification Report

## Verdict

**Related dataset, not the same dataset as the Minnie Ho and Jorge Troncoso paper.** The local README identifies the arrays as PEER Hub ImageNet (Phi-Net), Task 2: Damage State. The supplied arrays contain 13,271 labeled examples (11,811 train and 1,460 test), while the paper describes 5,913 labeled examples. We will use PEER Task 2 as the project dataset and label all results accordingly.

## Confirmed from local files

| File | Bytes | Shape | Dtype | Check
|---|---:|---|---|---|
| `dataset/task2_X_train.npy` | 7,111,544,960 | `(11811,224,224,3)` | float32 | Opens with memory mapping; exact header/size match; all values finite
| `dataset/task2_y_train.npy` | 94,616 | `(11811,2)` | float32 | Exact header/size match; valid one-hot; counts 6,282 / 5,529
| `dataset/task2_X_test.npy` | 879,083,648 | `(1460,224,224,3)` | float32 | Opens with memory mapping; exact header/size match; all values finite
| `dataset/task2_y_test.npy` | 11,808 | `(1460,2)` | float32 | Exact header/size match; valid one-hot; counts 745 / 715
| `dataset/README.txt` | 203 | text | - | Says Task 2 Damage State; index 0 Damaged, index 1 Undamaged
| `dataset/license.txt` | 20,849 | text | - | Present alongside the source data

Class balance from the README mapping:

- Training: 6,282 damaged (53.19%); 5,529 undamaged (46.81%).
- Test: 745 damaged (51.03%); 715 undamaged (48.97%).

The previous truncated test file is no longer present. The current test image file is 879,083,648 bytes and matches its NumPy header exactly.

## Image values and visual review

The images have the expected tensor dimensions and are finite. The global observed range is approximately `[-123.68, 151.061]`, so the stored values are not raw 8-bit RGB pixels. Adding `[103.939, 116.779, 123.68]` to sample channels gives exact integer-valued pixels. This strongly supports a Caffe-style transform: RGB-to-BGR channel order followed by subtraction of BGR means. This is an inference from the stored values; the experiment notebook reverses the operation before model-specific preprocessing.

Eight deterministic training examples were rendered in `reports/dataset_sample_preview.png`. They look plausible for the registered labels. A visual sample is a sanity check, not proof of every row pairing.

The local rerun confirmed finite values in all image rows and found zero exact duplicate groups among 13,271 rows using SHA-256 of each stored array row. No exact train/test duplicates were found. This cannot detect resized, cropped, or otherwise perceptually similar images; see `artifacts/local/exact_duplicate_scan.json`.

## Pairing and provenance limits

The label file row counts exactly match the associated image file counts, the files share conventional `X_train`/`y_train` and `X_test`/`y_test` names, and sampled image-label pairs look plausible. The local README gives the class mapping. However, it does not give a sample manifest, checksum list, or an explicit statement that each image row order is identical to the matching label row order. Treat row pairing as accepted based on the package naming/counts and sanity checks, with this residual limitation reported. No arrays were concatenated.

The current PEER download page places separate links named Task 2_1 and Task 2_2 under the single Task 2 Damage State heading. The page identifies Task 2 as the binary undamaged/damaged task, but the linked downloads are ZIP files that this browser could not inspect, and the short local README does not state whether 2_1 and 2_2 are train/test partitions or alternate releases. The local arrays already form conventional train/test pairs; we keep those pairs separate and do not combine files. The PEER report says the 2018 challenge release differs from the finalized benchmark. The exact release/version of the local arrays therefore remains unknown. Sources: [PEER Task 2 download listing](https://apps.peer.berkeley.edu/phi-net/download/), [Task 2 label definitions](https://apps.peer.berkeley.edu/phichallenge/?page_id=827), [PEER challenge/final benchmark distinction](https://peer.berkeley.edu/sites/default/files/publications/2019_07_final_1.9.2020_doi.pdf).

## Reference paper comparison

The reference report describes 5,913 labeled images: 3,186 undamaged and 2,727 damaged, with 224x224 8-bit RGB images. It reports a 90/10 training/validation split and says it did not set aside a separate test set. Table 1 gives KNN as 53.7%/52.9% under a combined Validation/Test heading; the other figures are single values. The layer-288 SVM prose explicitly says 75% validation. The report abstract calls InceptionV3's 83% a test accuracy, but this conflicts with the stated no-test protocol; a later section describes a separate 4,000-train/1,000-test diagnostic. The poster labels its table validation accuracy and gives MobileNetV1 67.7%, versus 67.0% in the report. Sources: [paper](https://cs229.stanford.edu/proj2018/report/39.pdf), [poster](https://cs229.stanford.edu/proj2018/poster/39.pdf), [authors' code](https://github.com/jatron/structural-damage-recognition), [PEER Phi-Net](https://apps.peer.berkeley.edu/phi-net/), [PEER report on challenge versus final benchmark](https://peer.berkeley.edu/sites/default/files/publications/2019_07_final_1.9.2020_doi.pdf). These reference values were not measured on the local arrays and must not be presented as our results.

The reference paper identifies PEER as its source and cites the 2018 PEER Hub ImageNet Challenge. The authors' repository contains image-sorting/splitting and model scripts, but its README does not give a dataset download instruction or include the 5,913-image dataset. PEER's current site lists Task 2_1 and Task 2_2 downloads under Damage State, but the PEER report says the 2018 challenge release differs from the finalized benchmark. The current download links therefore do not establish that the exact 5,913-image release is the same data as our local arrays. The repository contains multiple CNN variants; it is not one single frozen implementation. The project notebook documents deviations: provided PEER split, a stratified validation split from training, classical 32x32 resized features for resource use, frozen pretrained backbones, and current software implementations.

## Files and notebooks

- `notebooks/01_dataset_verification.ipynb`: header/size, dtype, shape, labels, pixel, and preview checks.
- `notebooks/02_experiments.ipynb`: configurable preprocessing, reference-model and ResNet50 training, metrics, and artifacts.
- `reports/FINAL_REPORT_TEMPLATE.md`: completed summary of the local results and reference-paper comparison.
