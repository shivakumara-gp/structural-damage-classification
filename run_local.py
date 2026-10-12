"""Local CPU/GPU runner for the PEER Task 2 structural damage project.

Start with: python run_local.py --stage classical
The input arrays are memory-mapped; the full 7.1 GB training array is never
copied into RAM. Outputs are written under artifacts/local/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
OUT = ROOT / "artifacts" / "local"
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)
CLASS_NAMES = ["Damaged", "Undamaged"]
SEED = 42


def verify_inputs():
    """Recheck header sizes, shapes, labels and finite pixels before training."""
    summary = {"data_dir": str(DATA), "files": {}, "labels": {}}
    specs = {
        "X_train": ((11811, 224, 224, 3), np.dtype("float32")),
        "X_test": ((1460, 224, 224, 3), np.dtype("float32")),
        "y_train": ((11811, 2), np.dtype("float32")),
        "y_test": ((1460, 2), np.dtype("float32")),
    }
    arrays = {}
    for key, (shape, dtype) in specs.items():
        path = DATA / f"task2_{key}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = int(np.prod(shape)) * dtype.itemsize + arr.offset
        if arr.shape != shape or arr.dtype != dtype or path.stat().st_size != expected:
            raise ValueError(f"Integrity mismatch for {path.name}: shape={arr.shape}, dtype={arr.dtype}, bytes={path.stat().st_size}, expected={expected}")
        summary["files"][path.name] = {"bytes": path.stat().st_size, "shape": arr.shape, "dtype": str(arr.dtype), "header_and_size_match": True}
        arrays[key] = arr
    # Scan in bounded chunks so the train tensor is never copied into memory.
    for key in ("X_train", "X_test"):
        arr = arrays[key]
        for start in range(0, len(arr), 128):
            if not np.isfinite(arr[start:start + 128]).all():
                raise ValueError(f"Non-finite image values found in {key}, starting at row {start}")
    for split in ("train", "test"):
        y = np.asarray(arrays[f"y_{split}"])
        if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all() or not np.all(y.sum(axis=1) == 1):
            raise ValueError(f"Invalid one-hot labels in {split}")
        summary["labels"][split] = {"damaged_index_0": int((y.argmax(1) == 0).sum()), "undamaged_index_1": int((y.argmax(1) == 1).sum())}
    # The source stores BGR pixels with channel-specific means removed. Restore,
    # then convert to RGB and use the reference paper's x/128-1 for classical models.
    summary["preprocessing"] = "Restore BGR means [103.939,116.779,123.68], reverse to RGB, resize 32x32, then x/128-1 for classical models. Mean interpretation was inferred from exact integer-valued restored samples."
    return arrays, summary


def scan_exact_duplicates(arrays):
    """SHA-256 every stored image row, streaming one at a time."""
    hashes = {}
    for split in ("train", "test"):
        rows = arrays[f"X_{split}"]
        for i in range(len(rows)):
            digest = hashlib.sha256(memoryview(np.asarray(rows[i])).cast("B")).hexdigest()
            hashes.setdefault(digest, []).append({"split": split, "row": i})
        print(f"Hashed {len(rows)} {split} image rows", flush=True)
    groups = [locations for locations in hashes.values() if len(locations) > 1]
    cross = [group for group in groups if len({v["split"] for v in group}) > 1]
    return {"method": "SHA-256 of the exact stored array row bytes", "duplicate_groups": groups,
            "cross_split_duplicate_groups": cross, "unique_hashes": len(hashes), "total_images": sum(len(arrays[f"X_{s}"]) for s in ("train", "test")),
            "scope_note": "Detects exact stored duplicates only, not resized/cropped/perceptual duplicates."}


def metrics(y, pred):
    cm = [[int(((y == i) & (pred == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    report = {}
    for cls, name in ((0, "Damaged"), (1, "Undamaged")):
        tp = cm[cls][cls]; fp = cm[1-cls][cls]; fn = cm[cls][1-cls]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        report[name] = {"precision": p, "recall": r, "f1-score": 2*p*r/(p+r) if p+r else 0.0, "support": int((y == cls).sum())}
    return {"accuracy": float((y == pred).mean()), "confusion_matrix_labels_0_damaged_1_undamaged": cm,
            "classification_report": report}


def knn_predict(X, y, Q, chunk=32, k=5):
    norms = np.einsum("ij,ij->i", X, X)
    pred = np.empty(len(Q), dtype=np.int64)
    for start in range(0, len(Q), chunk):
        q = np.asarray(Q[start:start+chunk], dtype=np.float32)
        d2 = np.einsum("ij,ij->i", q, q)[:, None] + norms[None, :] - 2 * (q @ X.T)
        nearest = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
        votes = (y[nearest] == 1).sum(axis=1)
        pred[start:start+len(q)] = (votes > k/2).astype(np.int64)
    return pred


def fit_logistic(X, y, Xval, yval, max_epochs=40, batch_size=512, lr=.001, C=1.0):
    """Mini-batch Adam for binary L2-regularized logistic loss; early stop on val loss."""
    n, d = X.shape; w = np.zeros(d, dtype=np.float64); b = 0.0
    mw=np.zeros_like(w); vw=np.zeros_like(w); mb=vb=0.0; eps=1e-8
    reg=1.0/(C*n); step=0; best=float("inf"); best_params=None; stale=0; rng=np.random.default_rng(SEED)
    def loss(features, targets):
        logits=np.clip(features@w+b,-40,40)
        return float(np.mean(np.logaddexp(0,logits)-targets*logits)+.5*reg*np.dot(w,w))
    for epoch in range(1,max_epochs+1):
        order=rng.permutation(n)
        for start in range(0,n,batch_size):
            ids=order[start:start+batch_size]; xb=np.asarray(X[ids],dtype=np.float64); yb=y[ids]
            logits=np.clip(xb@w+b,-40,40); probs=1/(1+np.exp(-logits)); err=probs-yb
            gw=(xb.T@err)/len(ids)+reg*w; gb=float(err.mean()); step+=1
            mw=.9*mw+.1*gw; vw=.999*vw+.001*gw*gw
            mb=.9*mb+.1*gb; vb=.999*vb+.001*gb*gb
            w-=lr*(mw/(1-.9**step))/(np.sqrt(vw/(1-.999**step))+eps)
            b-=lr*(mb/(1-.9**step))/(np.sqrt(vb/(1-.999**step))+eps)
        current=loss(Xval,yval)
        if current < best-1e-6:
            best=current; best_params=(w.copy(),b); stale=0
        else:
            stale+=1
            if stale>=8: break
    w,b=best_params
    return {"weights":w,"bias":float(b),"epochs_completed":epoch,"optimizer_steps":step,
            "best_validation_loss":best,"early_stopped":stale>=8,"learning_rate":lr,"C":C}


def fit_rbf_smo(X, y, C=1.0, gamma=.001, max_passes=8, max_updates=30000, seed=42):
    """Small-set binary RBF SVM trained with a NumPy simplified SMO solver."""
    n=len(y); target=np.where(y==1,1.0,-1.0); norms=np.einsum("ij,ij->i",X,X)
    gram=X@X.T; dist=np.maximum(norms[:,None]+norms[None,:]-2*gram,0)
    K=np.exp(-gamma*dist).astype(np.float64,copy=False)
    alpha=np.zeros(n,dtype=np.float64); bias=0.0; error=-target.copy()
    rng=np.random.default_rng(seed); unchanged=0; updates=0
    while unchanged < max_passes and updates < max_updates:
        changed=0
        for i in range(n):
            ei=error[i]; yi=target[i]; ai=alpha[i]
            if not ((yi*ei < -1e-3 and ai < C-1e-8) or (yi*ei > 1e-3 and ai > 1e-8)): continue
            candidates=np.arange(n); candidates=candidates[candidates!=i]
            nonbound=candidates[(alpha[candidates]>1e-8)&(alpha[candidates]<C-1e-8)]
            pool=nonbound if len(nonbound) else candidates
            j=int(pool[np.argmax(np.abs(ei-error[pool]))]) if len(nonbound) else int(rng.choice(pool))
            ej=error[j]; yj=target[j]; aj=alpha[j]
            if yi != yj: lo=max(0.0,aj-ai); hi=min(C,C+aj-ai)
            else: lo=max(0.0,ai+aj-C); hi=min(C,ai+aj)
            if hi-lo < 1e-12: continue
            eta=2*K[i,j]-K[i,i]-K[j,j]
            if eta >= -1e-12: continue
            new_aj=np.clip(aj-yj*(ei-ej)/eta,lo,hi)
            if abs(new_aj-aj)<1e-7: continue
            new_ai=ai+yi*yj*(aj-new_aj)
            old_b=bias
            b1=bias-ei-yi*(new_ai-ai)*K[i,i]-yj*(new_aj-aj)*K[i,j]
            b2=bias-ej-yi*(new_ai-ai)*K[i,j]-yj*(new_aj-aj)*K[j,j]
            bias=b1 if 1e-8<new_ai<C-1e-8 else (b2 if 1e-8<new_aj<C-1e-8 else (b1+b2)/2)
            di=yi*(new_ai-ai); dj=yj*(new_aj-aj)
            alpha[i]=new_ai; alpha[j]=new_aj
            error += di*K[:,i]+dj*K[:,j]+(bias-old_b)
            changed+=1; updates+=1
            if updates>=max_updates: break
        unchanged=unchanged+1 if changed==0 else 0
        if unchanged % 2 == 0 and unchanged: print(f"  SVM SMO sweep no-change count {unchanged}; updates={updates}",flush=True)
    support=alpha>1e-7
    return {"support_vectors":X[support].copy(),"dual_coef":(alpha[support]*target[support]).copy(),"bias":float(bias),
            "gamma":gamma,"C":C,"solver":"NumPy SMO","updates":updates,"support_count":int(support.sum()),"converged_passes":unchanged}


def svm_predict(model, Q, chunk=128):
    sv=model["support_vectors"]; coef=model["dual_coef"]; gamma=model["gamma"]
    svnorm=np.einsum("ij,ij->i",sv,sv); pred=np.empty(len(Q),dtype=np.int64)
    for start in range(0,len(Q),chunk):
        q=np.asarray(Q[start:start+chunk],dtype=np.float32); qnorm=np.einsum("ij,ij->i",q,q)
        dist=np.maximum(qnorm[:,None]+svnorm[None,:]-2*(q@sv.T),0)
        pred[start:start+len(q)]=((np.exp(-gamma*dist)@coef+model["bias"])>=0).astype(np.int64)
    return pred


def make_features(arr, destination: Path, chunk=64):
    """Stream an NCHW-independent 32x32 RGB float32 feature file from the source."""
    if destination.exists() and destination.stat().st_size == arr.shape[0] * 32 * 32 * 3 * 4:
        return np.load(destination, mmap_mode="r", allow_pickle=False)
    out = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=(len(arr), 32, 32, 3))
    for start in range(0, len(arr), chunk):
        stop = min(start + chunk, len(arr))
        restored_bgr = np.asarray(arr[start:stop], dtype=np.float32) + MEAN_BGR
        rgb = np.clip(restored_bgr[..., ::-1], 0, 255).astype(np.uint8)
        for j, image in enumerate(rgb):
            small = Image.fromarray(image, mode="RGB").resize((32, 32), Image.Resampling.BILINEAR)
            out[start + j] = np.asarray(small, dtype=np.float32) / 128.0 - 1.0
    out.flush()
    return np.load(destination, mmap_mode="r", allow_pickle=False)


def save_classical_results(arrays, inventory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    x_train_path, x_test_path = OUT / "X_train_32_rgb.npy", OUT / "X_test_32_rgb.npy"
    X = make_features(arrays["X_train"], x_train_path).reshape(len(arrays["X_train"]), -1)
    Xtest = make_features(arrays["X_test"], x_test_path).reshape(len(arrays["X_test"]), -1)
    y = arrays["y_train"].argmax(1)
    ytest = arrays["y_test"].argmax(1)
    rng=np.random.default_rng(SEED); fit_idx=[]; val_idx=[]
    for cls in (0,1):
        ids=np.flatnonzero(y==cls); rng.shuffle(ids); nval=max(1,int(round(.10*len(ids))))
        val_idx.extend(ids[:nval]); fit_idx.extend(ids[nval:])
    fit_idx=np.asarray(fit_idx); val_idx=np.asarray(val_idx)
    # SVC's O(n^2) kernel makes fitting all 10.6k rows memory/time intensive.
    # Preserve the prescribed C/gamma, but run a seeded 2,500-row fit subset and
    # clearly record this compute adaptation. KNN and logistic use all fit rows.
    y_binary=(y == 1).astype(np.float64)
    svm_idx=np.random.default_rng(SEED).choice(fit_idx,min(2500,len(fit_idx)),replace=False)
    models = [("KNN_k5", "knn"), ("LogisticRegression", "logistic"), ("RBF_SVM", "svm")]
    rows = []
    reports = {}
    for name, kind in models:
        use_idx=svm_idx if kind=="svm" else fit_idx
        t0 = time.perf_counter()
        if kind=="logistic": model=fit_logistic(X[fit_idx],y_binary[fit_idx],X[val_idx],y_binary[val_idx],max_epochs=40,batch_size=512,lr=.001,C=1.0)
        elif kind=="svm": model=fit_rbf_smo(X[svm_idx],y[svm_idx],C=1.0,gamma=.001,seed=SEED)
        else: model=(np.asarray(X[fit_idx]),y[fit_idx])
        train_s = time.perf_counter() - t0
        def evaluate(features, labels):
            t1 = time.perf_counter()
            if kind=="knn": pred=knn_predict(model[0],model[1],features)
            elif kind=="logistic": pred=((np.asarray(features)@model["weights"]+model["bias"])>=0).astype(np.int64)
            else: pred=svm_predict(model,features)
            elapsed = time.perf_counter() - t1
            result=metrics(labels,pred)
            return {**result,
                    "inference_ms_per_image": elapsed * 1000 / len(labels), "predictions": pred.tolist()}
        val = evaluate(X[val_idx], y[val_idx])
        test = evaluate(Xtest, ytest)
        test_pred = np.asarray(test.pop("predictions"), dtype=np.int64)
        params=({"k":5} if kind=="knn" else {"C":1,"penalty":"L2","optimizer":"mini-batch Adam",**{k:v for k,v in model.items() if k not in ("weights",)}} if kind=="logistic" else {"C":1,"gamma":.001,"kernel":"RBF",**{k:v for k,v in model.items() if k not in ("support_vectors","dual_coef")}})
        row = {"model": name, "features": "32x32 RGB flattened; x/128-1", "hyperparameters": params,
               "validation_accuracy": val["accuracy"], "test_accuracy": test["accuracy"], "test_precision_damaged": test["classification_report"]["Damaged"]["precision"],
               "test_recall_damaged": test["classification_report"]["Damaged"]["recall"], "test_f1_damaged": test["classification_report"]["Damaged"]["f1-score"],
               "train_seconds": train_s, "inference_ms_per_image_model_only": test["inference_ms_per_image"], "train_rows": len(use_idx), "result_type": "local experiment",
               "solver_note": ("NumPy exact nearest-neighbor search; k=5." if kind=="knn" else
                              "NumPy mini-batch Adam, L2 penalty C=1, validation-loss early stopping; not scikit-learn LBFGS." if kind=="logistic" else
                              "NumPy SMO RBF SVM, C=1, gamma=0.001; fit on a seeded 2,500-row subset." )}
        rows.append(row)
        reports[name] = {"validation": val, "test": test}
        if kind=="logistic": np.savez(OUT/f"{name}.npz",weights=model["weights"],bias=model["bias"])
        elif kind=="svm": np.savez(OUT/f"{name}.npz",**model)
        else: np.savez(OUT/f"{name}.npz",support_features=model[0],support_labels=model[1])
        np.save(OUT / f"{name}_test_predictions.npy", test_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(test["confusion_matrix_labels_0_damaged_1_undamaged"], cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Damaged", "Undamaged"], yticklabels=["Damaged", "Undamaged"], xlabel="Predicted", ylabel="Actual", title=f"{name} test confusion matrix")
        for (i, j), value in np.ndenumerate(test["confusion_matrix_labels_0_damaged_1_undamaged"]): ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout(); fig.savefig(OUT / f"{name}_confusion_matrix.png", dpi=160); plt.close(fig)
        for kind, candidates in (("correct", np.flatnonzero(test_pred == ytest)), ("wrong", np.flatnonzero(test_pred != ytest))):
            chosen = candidates[:8]
            if not len(chosen): continue
            fig, axes = plt.subplots(1, len(chosen), figsize=(2.5 * len(chosen), 3))
            axes = np.atleast_1d(axes)
            for ax, idx in zip(axes, chosen):
                bgr = np.asarray(arrays["X_test"][int(idx)]) + MEAN_BGR
                rgb = np.clip(bgr[..., ::-1], 0, 255).astype(np.uint8)
                ax.imshow(rgb); ax.set_title(f"true={CLASS_NAMES[int(ytest[idx])]}\npred={CLASS_NAMES[int(test_pred[idx])]}", fontsize=8); ax.axis("off")
            fig.tight_layout(); fig.savefig(OUT / f"{name}_test_{kind}_examples.png", dpi=130); plt.close(fig)
        print(f"{name}: val={val['accuracy']:.4f}, test={test['accuracy']:.4f}, fit={train_s:.1f}s")
    (OUT / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (OUT / "classical_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (OUT / "classical_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    import csv
    with (OUT / "classical_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar([r["model"] for r in rows], [r["test_accuracy"] for r in rows]); ax.set(ylabel="Test accuracy", ylim=(0, 1), title="Classical model test accuracy"); ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(OUT / "classical_accuracy_comparison.png", dpi=160); plt.close(fig)
    run_meta = {"python": platform.python_version(), "platform": platform.platform(), "seed": SEED, "validation_fraction": .10, "dataset": inventory, "stage": "classical"}
    (OUT / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["verify", "classical", "deep"], default="verify")
    args = parser.parse_args()
    arrays, inventory = verify_inputs()
    print("Dataset verification passed:", inventory["labels"])
    if args.stage == "verify":
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        duplicate_report = scan_exact_duplicates(arrays)
        (OUT / "exact_duplicate_scan.json").write_text(json.dumps(duplicate_report, indent=2), encoding="utf-8")
        print("Exact duplicate groups:", len(duplicate_report["duplicate_groups"]), "cross-split:", len(duplicate_report["cross_split_duplicate_groups"]))
    elif args.stage == "classical":
        save_classical_results(arrays, inventory)
    else:
        raise SystemExit("Run the local transfer-learning runner with: python run_local_deep.py")


if __name__ == "__main__":
    main()
