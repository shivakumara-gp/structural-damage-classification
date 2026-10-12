"""Load a saved local model and classify one image file."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"artifacts"/"local"
CLASSES=["Damaged","Undamaged"]

def load_rgb(image_path):
    image=Image.open(image_path).convert("RGB").resize((32,32),Image.Resampling.BILINEAR)
    rgb=np.asarray(image,dtype=np.float32)
    return rgb, (rgb/128.0-1.0).reshape(1,-1).astype(np.float32)

def predict(model_name, rgb, flat):
    if model_name=="knn":
        saved=np.load(OUT/"KNN_k5.npz",allow_pickle=False)
        X=saved["support_features"]; y=saved["support_labels"]
        norm=np.einsum("ij,ij->i",X,X)
        d2=np.einsum("ij,ij->i",flat,flat)[:,None]+norm[None,:]-2*(flat@X.T)
        ids=np.argpartition(d2[0],4)[:5]
        return int((y[ids]==1).sum()>2), "KNN k=5"
    if model_name=="logistic":
        saved=np.load(OUT/"LogisticRegression.npz",allow_pickle=False)
        return int((flat@saved["weights"]+float(saved["bias"]))[0]>=0), "Logistic regression"
    if model_name=="svm":
        saved=np.load(OUT/"RBF_SVM.npz",allow_pickle=False)
        sv=saved["support_vectors"]; norms=np.einsum("ij,ij->i",sv,sv)
        dist=np.maximum(np.einsum("ij,ij->i",flat,flat)[:,None]+norms[None,:]-2*(flat@sv.T),0)
        decision=np.exp(-float(saved["gamma"])*dist)@saved["dual_coef"]+float(saved["bias"])
        return int(decision[0]>=0), "RBF-SVM"
    if model_name=="cnn":
        from run_numpy_cnn import forward
        saved=np.load(OUT/"NumpyTinyCNN.npz",allow_pickle=False)
        params={key:saved[key] for key in saved.files}
        # Training converted x/128-1 features with *0.5+0.5, so it used RGB/256.
        x=np.expand_dims(rgb/256.0,axis=0).astype(np.float32)
        logits=forward(x,params)
        return int(logits.argmax(axis=1)[0]), "NumPy Tiny CNN"
    raise ValueError(model_name)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",type=Path,help="Path to an RGB/JPEG/PNG image")
    parser.add_argument("--model",choices=["all","knn","logistic","svm","cnn"],default="all")
    args=parser.parse_args()
    if not args.image.is_file(): raise SystemExit(f"Image not found: {args.image}")
    rgb,flat=load_rgb(args.image)
    names=["knn","logistic","svm","cnn"] if args.model=="all" else [args.model]
    for name in names:
        class_id,label=predict(name,rgb,flat)
        print(f"{label}: {CLASSES[class_id]}")

if __name__=="__main__": main()
