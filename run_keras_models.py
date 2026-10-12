"""Run ImageNet Keras backbones locally on CPU and record PEER Task 2 metrics.

Frozen pretrained backbones are used as feature extractors, followed by a trained
linear head. The Inception layer-288 SVM uses spatially pooled layer activations
and a seeded 2,500-row training subset to keep the RBF kernel tractable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
OUT = ROOT / "artifacts" / "local"
SEED = 42
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)
CLASS_NAMES = ["Damaged", "Undamaged"]
BATCH_SIZE = 8
EPOCHS = 30


def split_indices(labels):
    rng = np.random.default_rng(SEED)
    fit, val = [], []
    for cls in (0, 1):
        ids = np.flatnonzero(labels == cls)
        rng.shuffle(ids)
        nval = max(1, int(round(0.10 * len(ids))))
        val.extend(ids[:nval])
        fit.extend(ids[nval:])
    return np.asarray(fit, dtype=np.int64), np.asarray(val, dtype=np.int64)


def make_backbone(name):
    if name == "mobilenet_v1":
        base = tf.keras.applications.MobileNet(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
        )
        prep = tf.keras.applications.mobilenet.preprocess_input
        return base, prep, 224, "MobileNetV1 (Keras ImageNet weights, alpha=1.0)"
    if name == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
        )
        prep = tf.keras.applications.resnet50.preprocess_input
        return base, prep, 224, "ResNet50 (Keras ImageNet weights)"
    raise ValueError(name)


def feature_dataset(images, ids, image_size, preprocess):
    ids = np.asarray(ids, dtype=np.int64)

    def gen():
        for idx in ids:
            yield np.asarray(images[int(idx)], dtype=np.float32)

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=tf.TensorSpec((224, 224, 3), tf.float32)
    )

    def transform(stored):
        bgr = tf.clip_by_value(stored + tf.constant(MEAN_BGR), 0.0, 255.0)
        rgb = tf.reverse(bgr, axis=[-1])
        if image_size != 224:
            rgb = tf.image.resize(rgb, (image_size, image_size), antialias=True)
        return preprocess(rgb)

    return ds.map(transform, num_parallel_calls=2).batch(BATCH_SIZE).prefetch(1)


def cache_features(model, images, ids, image_size, preprocess, cache_path, expected_dim):
    n = len(ids)
    if cache_path.exists():
        try:
            cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if cached.shape == (n, expected_dim) and cached.dtype == np.float32:
                print(f"  reuse {cache_path.name}: {cached.shape}", flush=True)
                return cached, 0.0
        except Exception:
            pass
    out = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=(n, expected_dim)
    )
    ds = feature_dataset(images, ids, image_size, preprocess)
    cursor = 0
    started = time.perf_counter()
    for batch in ds:
        z = model(batch, training=False)
        if isinstance(z, (tuple, list)):
            z = z[0]
        z = np.asarray(z.numpy(), dtype=np.float32).reshape(len(batch), -1)
        out[cursor:cursor + len(z)] = z
        cursor += len(z)
        if cursor % 512 < BATCH_SIZE or cursor == n:
            print(f"  features {cache_path.stem}: {cursor}/{n}", flush=True)
    out.flush()
    del out
    print(f"  extracted {n} rows in {time.perf_counter()-started:.1f}s", flush=True)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False), time.perf_counter()-started


def train_head(name, model, image_size, preprocess, xtr, ytr, xte, yte, fit_ids, val_ids):
    dims = int(model.output_shape[-1])
    spec = model.name
    total_extract = 0.0
    features = {}
    for split, images, ids in (("fit", xtr, fit_ids), ("val", xtr, val_ids), ("test", xte, np.arange(len(yte)))):
        p = OUT / f"{name}_{split}_pooled_features.npy"
        features[split], elapsed = cache_features(model, images, ids, image_size, preprocess, p, dims)
        total_extract += elapsed

    inputs = tf.keras.Input(shape=(dims,), name="imagenet_features")
    outputs = tf.keras.layers.Dense(2, name="damage_logits")(inputs)
    head = tf.keras.Model(inputs, outputs, name=f"{name}_linear_head")
    head.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    fit_started = time.perf_counter()
    history = head.fit(
        np.asarray(features["fit"]), ytr[fit_ids],
        validation_data=(np.asarray(features["val"]), ytr[val_ids]),
        batch_size=256, epochs=EPOCHS, shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    head_seconds = time.perf_counter() - fit_started
    print(f"  linear head fit complete: {len(history.history['loss'])} epochs, {head_seconds:.1f}s", flush=True)
    val_logits = head.predict(np.asarray(features["val"]), verbose=0)
    t0 = time.perf_counter()
    test_logits = head.predict(np.asarray(features["test"]), verbose=0)
    infer_seconds = time.perf_counter() - t0
    val_pred = val_logits.argmax(axis=1)
    test_pred = test_logits.argmax(axis=1)
    print("  validation and test predictions complete", flush=True)
    result = make_result(
        name=name, val_true=ytr[val_ids], val_pred=val_pred, test_true=yte, test_pred=test_pred,
        features="ImageNet-pretrained frozen Keras backbone, global average pooled features",
        config={"backbone": spec, "weights": "ImageNet", "backbone_trainable": False,
                "head": "Dense(2), SGD learning_rate=0.01", "epochs_run": len(history.history["loss"]),
                "batch_size_features": BATCH_SIZE, "seed": SEED, "input_size": image_size},
        train_seconds=total_extract + head_seconds,
        infer_ms=1000.0 * infer_seconds / len(yte),
        extra={"feature_extraction_seconds": total_extract, "head_fit_seconds": head_seconds,
               "feature_dimension": dims, "weights": "ImageNet", "framework": f"TensorFlow {tf.__version__}"},
    )
    np.save(OUT / f"{name}_test_predictions.npy", test_pred)
    (OUT / f"{name}_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    np.savez(OUT / f"{name}_linear_head.npz", kernel=head.get_weights()[0], bias=head.get_weights()[1])
    print("  saved linear head and training history", flush=True)
    write_confusions(name, ytr[val_ids], val_pred, yte, test_pred)
    print("  saved confusion matrices", flush=True)
    append_result(result)
    print("  updated combined results", flush=True)
    print(f"{name}: validation={result['validation_accuracy']:.4f}, test={result['test_accuracy']:.4f}", flush=True)
    del features, head
    return result


def inception_pair(xtr, ytr, xte, yte, fit_ids, val_ids):
    full = tf.keras.applications.InceptionV3(
        include_top=True, weights="imagenet", input_shape=(299, 299, 3)
    )
    if len(full.layers) <= 288:
        raise RuntimeError(f"Keras InceptionV3 has only {len(full.layers)} layers")
    chosen = full.layers[288]
    print(f"InceptionV3 layers={len(full.layers)}; layer 288={chosen.name}, shape={chosen.output.shape}", flush=True)
    avg_layer = full.get_layer("avg_pool")
    multi = tf.keras.Model(full.input, [avg_layer.output, chosen.output], name="inception_features")
    preprocess = tf.keras.applications.inception_v3.preprocess_input
    paths = {
        "fit": (xtr, fit_ids), "val": (xtr, val_ids), "test": (xte, np.arange(len(yte)))
    }
    caches = {}
    dims_base = int(avg_layer.output.shape[-1])
    dims_288 = int(chosen.output.shape[-1])
    extract_seconds = 0.0
    for split, (images, ids) in paths.items():
        base_path = OUT / f"inception_v3_{split}_pooled_features.npy"
        layer_path = OUT / f"inception_layer288_{split}_pooled_features.npy"
        ids = np.asarray(ids, dtype=np.int64)
        base_ok = valid_cache(base_path, len(ids), dims_base)
        layer_ok = valid_cache(layer_path, len(ids), dims_288)
        if base_ok and layer_ok:
            caches[(split, "base")] = np.load(base_path, mmap_mode="r", allow_pickle=False)
            caches[(split, "layer")]= np.load(layer_path, mmap_mode="r", allow_pickle=False)
            print(f"  reuse Inception {split} feature caches", flush=True)
            continue
        base_out = None if base_ok else np.lib.format.open_memmap(base_path, mode="w+", dtype=np.float32, shape=(len(ids), dims_base))
        layer_out = None if layer_ok else np.lib.format.open_memmap(layer_path, mode="w+", dtype=np.float32, shape=(len(ids), dims_288))
        ds = feature_dataset(images, ids, 299, preprocess)
        cursor = 0
        t0 = time.perf_counter()
        for batch in ds:
            pooled, activation = multi(batch, training=False)
            if base_out is not None:
                base_out[cursor:cursor+len(batch)] = pooled.numpy().astype(np.float32)
            if layer_out is not None:
                layer_out[cursor:cursor+len(batch)] = tf.reduce_mean(activation, axis=(1, 2)).numpy().astype(np.float32)
            cursor += len(batch)
            if cursor % 512 < BATCH_SIZE or cursor == len(ids):
                print(f"  Inception {split}: {cursor}/{len(ids)}", flush=True)
        if base_out is not None:
            base_out.flush(); del base_out
        if layer_out is not None:
            layer_out.flush(); del layer_out
        elapsed = time.perf_counter() - t0
        extract_seconds += elapsed
        caches[(split, "base")] = np.load(base_path, mmap_mode="r", allow_pickle=False)
        caches[(split, "layer")] = np.load(layer_path, mmap_mode="r", allow_pickle=False)
        print(f"  Inception {split} features extracted in {elapsed:.1f}s", flush=True)
    base_extract = extract_seconds
    # Reuse the feature-head fit path with the already generated pooled embeddings.
    base_result = train_head_from_cache(
        "inception_v3", caches, "base", "InceptionV3 (Keras ImageNet weights; frozen; avg_pool)",
        ytr, yte, fit_ids, val_ids, total_extract=base_extract,
        config_extra={"backbone_trainable": False, "input_size": 299, "layers": len(full.layers)},
    )
    layer_result = run_layer_svm(
        caches, ytr, yte, fit_ids, val_ids, chosen.name, tuple(chosen.output.shape[1:]),
        total_extract=base_extract,
    )
    del full, multi, caches
    tf.keras.backend.clear_session()
    return base_result, layer_result


def valid_cache(path, n, d):
    if not path.exists():
        return False
    try:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        return data.shape == (n, d) and data.dtype == np.float32
    except Exception:
        return False


def train_head_from_cache(name, caches, tag, backbone, ytr, yte, fit_ids, val_ids, total_extract, config_extra):
    dims = caches[("fit", tag)].shape[1]
    inp = tf.keras.Input(shape=(dims,), name="imagenet_features")
    out = tf.keras.layers.Dense(2, name="damage_logits")(inp)
    head = tf.keras.Model(inp, out, name=f"{name}_linear_head")
    head.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    t0 = time.perf_counter()
    history = head.fit(np.asarray(caches[("fit", tag)]), ytr[fit_ids],
                       validation_data=(np.asarray(caches[("val", tag)]), ytr[val_ids]),
                       batch_size=256, epochs=EPOCHS, shuffle=True, verbose=0,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    head_seconds = time.perf_counter() - t0
    print(f"  Inception linear head fit complete: {len(history.history['loss'])} epochs, {head_seconds:.1f}s", flush=True)
    val_pred = head.predict(np.asarray(caches[("val", tag)]), verbose=0).argmax(axis=1)
    t0 = time.perf_counter()
    test_pred = head.predict(np.asarray(caches[("test", tag)]), verbose=0).argmax(axis=1)
    print("  Inception validation and test predictions complete", flush=True)
    infer = time.perf_counter() - t0
    result = make_result(name, ytr[val_ids], val_pred, yte, test_pred,
                         "ImageNet-pretrained frozen InceptionV3, global average pooled avg_pool features",
                         {"backbone": backbone, "weights": "ImageNet", "head": "Dense(2), SGD lr=0.01",
                          "epochs_run": len(history.history["loss"]), "batch_size_features": BATCH_SIZE,
                          "seed": SEED, **config_extra},
                         total_extract + head_seconds, 1000.0 * infer / len(yte),
                         {"feature_extraction_seconds": total_extract, "head_fit_seconds": head_seconds,
                          "feature_dimension": dims, "framework": f"TensorFlow {tf.__version__}"})
    np.save(OUT / f"{name}_test_predictions.npy", test_pred)
    (OUT / f"{name}_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    np.savez(OUT / f"{name}_linear_head.npz", kernel=head.get_weights()[0], bias=head.get_weights()[1])
    print("  saved Inception linear head and history", flush=True)
    write_confusions(name, ytr[val_ids], val_pred, yte, test_pred)
    print("  saved Inception confusion matrices", flush=True)
    append_result(result)
    print("  updated combined results", flush=True)
    print(f"{name}: validation={result['validation_accuracy']:.4f}, test={result['test_accuracy']:.4f}", flush=True)
    del head
    return result


def run_layer_svm(caches, ytr, yte, fit_ids, val_ids, layer_name, activation_shape, total_extract):
    from run_local import fit_rbf_smo, svm_predict
    x_fit = np.asarray(caches[("fit", "layer")], dtype=np.float32)
    x_val = np.asarray(caches[("val", "layer")], dtype=np.float32)
    x_test = np.asarray(caches[("test", "layer")], dtype=np.float32)
    subset = np.random.default_rng(SEED).choice(np.arange(len(fit_ids)), min(2500, len(fit_ids)), replace=False)
    y_fit = ytr[fit_ids][subset]
    t0 = time.perf_counter()
    svm = fit_rbf_smo(x_fit[subset], y_fit, C=1.0, gamma=0.001, seed=SEED)
    fit_seconds = time.perf_counter() - t0
    val_pred = svm_predict(svm, x_val)
    t0 = time.perf_counter()
    test_pred = svm_predict(svm, x_test)
    infer = time.perf_counter() - t0
    result = make_result(
        "InceptionV3_layer288_RBF_SVM", ytr[val_ids], val_pred, yte, test_pred,
        f"Keras InceptionV3 layer 288 ({layer_name}) activation, spatial mean pooled to {x_fit.shape[1]} features",
        {"architecture": "Keras InceptionV3 include_top=True, ImageNet weights", "layer_index": 288,
         "layer_name": layer_name, "layer_activation_shape": list(activation_shape), "pooling": "spatial mean",
         "kernel": "RBF", "C": 1.0, "gamma": 0.001, "solver": "NumPy SMO",
         "fit_rows": len(subset), "fit_subset_seed": SEED, "support_count": svm["support_count"],
         "solver_updates": svm["updates"], "input_size": 299},
        total_extract + fit_seconds, 1000.0 * infer / len(yte),
        {"feature_extraction_seconds": total_extract, "svm_fit_seconds": fit_seconds,
         "feature_dimension": x_fit.shape[1], "framework": f"TensorFlow {tf.__version__}",
         "implementation_note": "Spatial pooling and seeded 2,500-row subset are compute adaptations."},
    )
    np.save(OUT / "inception_layer288_svm_test_predictions.npy", test_pred)
    np.savez(OUT / "inception_layer288_svm_model.npz", **svm)
    write_confusions("InceptionV3_layer288_RBF_SVM", ytr[val_ids], val_pred, yte, test_pred)
    append_result(result)
    print(f"InceptionV3 layer-288 SVM: validation={result['validation_accuracy']:.4f}, test={result['test_accuracy']:.4f}", flush=True)
    return result


def make_result(name, val_true, val_pred, test_true, test_pred, features, config, train_seconds, infer_ms, extra=None):
    cm = confusion(test_true, test_pred)
    tp = cm[0][0]
    precision = tp / max(1, cm[0][0] + cm[1][0])
    recall = tp / max(1, cm[0][0] + cm[0][1])
    f1 = 2 * precision * recall / max(1e-15, precision + recall)
    out = {
        "model": name, "features": features, "hyperparameters": config,
        "validation_accuracy": float(np.mean(np.asarray(val_true) == np.asarray(val_pred))),
        "test_accuracy": float(np.mean(np.asarray(test_true) == np.asarray(test_pred))),
        "test_precision_damaged": float(precision), "test_recall_damaged": float(recall),
        "test_f1_damaged": float(f1), "test_confusion_matrix_labels_0_damaged_1_undamaged": cm,
        "train_seconds": float(train_seconds), "inference_ms_per_image_model_only": float(infer_ms),
        "result_type": "local experiment",
    }
    out.update(extra or {})
    return out


def confusion(truth, pred):
    truth = np.asarray(truth); pred = np.asarray(pred)
    return [[int(np.sum((truth == i) & (pred == j))) for j in (0, 1)] for i in (0, 1)]


def write_confusions(name, yv, pv, yt, pt):
    for split, truth, pred in (("validation", yv, pv), ("test", yt, pt)):
        cm = confusion(truth, pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="Predicted", ylabel="Actual", title=f"{name} {split}")
        for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha="center", va="center")
        fig.tight_layout(); fig.savefig(OUT / f"{name}_{split}_confusion.png", dpi=160)
        plt.close(fig)


def append_result(result):
    safe_name = result["model"].lower().replace(" ", "_").replace("/", "_")
    (OUT / f"{safe_name}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    p = OUT / "all_model_results.json"
    rows = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    rows = [r for r in rows if r.get("model") != result["model"]]
    rows.append(result)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["model", "validation_accuracy", "test_accuracy", "test_precision_damaged",
              "test_recall_damaged", "test_f1_damaged", "train_seconds", "inference_ms_per_image_model_only"]
    with (OUT / "all_model_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["mobilenet_v1", "inception_pair", "resnet50"],
                        choices=["mobilenet_v1", "inception_pair", "resnet50"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    xtr = np.load(DATA / "task2_X_train.npy", mmap_mode="r", allow_pickle=False)
    ytr = np.load(DATA / "task2_y_train.npy", mmap_mode="r", allow_pickle=False).argmax(axis=1).astype(np.int64)
    xte = np.load(DATA / "task2_X_test.npy", mmap_mode="r", allow_pickle=False)
    yte = np.load(DATA / "task2_y_test.npy", mmap_mode="r", allow_pickle=False).argmax(axis=1).astype(np.int64)
    fit_ids, val_ids = split_indices(ytr)
    print(f"TensorFlow {tf.__version__}; GPU devices={tf.config.list_physical_devices('GPU')}; train={len(fit_ids)} val={len(val_ids)} test={len(yte)}", flush=True)
    for name in args.models:
        if name == "inception_pair":
            inception_pair(xtr, ytr, xte, yte, fit_ids, val_ids)
            continue
        model, preprocess, image_size, spec = make_backbone(name)
        print(f"Loaded {spec}; pooled features={model.output_shape[-1]}", flush=True)
        train_head(name, model, image_size, preprocess, xtr, ytr, xte, yte, fit_ids, val_ids)
        del model
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
