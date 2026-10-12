"""GPU transfer-learning runs on the local PEER Task 2 arrays.

The large input arrays stay memory-mapped. Frozen ImageNet backbones produce
cached pooled features; a small classifier head is selected on validation data.
Run: python run_local_deep.py [--models mobilenet_v1 inception_v3 resnet50]
"""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"
OUT=ROOT/"artifacts"/"local"
SEED=42
MEAN_BGR=np.array([103.939,116.779,123.68],dtype=np.float32)
CLASS_NAMES=["Damaged","Undamaged"]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

def make_backbone(name, device):
    if name == "mobilenet_v1":
        import timm
        options=timm.list_models("mobilenetv1*")
        model_name=next((n for n in options if "ra4" in n), None) or (options[0] if options else None)
        if not model_name: raise RuntimeError("Installed timm does not provide a MobileNetV1 model")
        model=timm.create_model(model_name,pretrained=True,num_classes=0)
        model.eval(); model.to(device)
        return model, model_name
    from torchvision import models
    if name == "resnet50":
        weights=models.ResNet50_Weights.DEFAULT
        model=models.resnet50(weights=weights)
        model.fc=nn.Identity()
        return model.eval().to(device), "torchvision ResNet50_Weights.DEFAULT"
    if name == "inception_v3":
        weights=models.Inception_V3_Weights.DEFAULT
        model=models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
        model.aux_logits=False
        model.fc=nn.Identity()
        return model.eval().to(device), "torchvision Inception_V3_Weights.DEFAULT"
    raise ValueError(name)

def preprocess_rows(arr, ids, size):
    # Input array is already memory mapped. Convert source BGR mean-centered
    # pixels back to RGB, resize, and apply ImageNet channel normalization.
    block=np.asarray(arr[ids],dtype=np.float32)+MEAN_BGR
    rgb=np.clip(block[...,::-1],0,255)/255.0
    tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(0,3,1,2)))
    tensor=torch.nn.functional.interpolate(tensor,size=(size,size),mode="bilinear",align_corners=False,antialias=True)
    mean=torch.tensor([.485,.456,.406],dtype=tensor.dtype).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],dtype=tensor.dtype).view(1,3,1,1)
    return (tensor-mean)/std

def extract_features(backbone,name,arr,ids,size,batch,device,outpath):
    if outpath.exists():
        return np.load(outpath,mmap_mode="r",allow_pickle=False),0.0
    feats=None; cursor=0; started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(ids),batch):
            sel=ids[start:start+batch]
            x=preprocess_rows(arr,sel,size).to(device)
            z=backbone(x)
            if isinstance(z,tuple): z=z[0]
            if z.ndim>2: z=torch.nn.functional.adaptive_avg_pool2d(z,1).flatten(1)
            z=z.float().cpu().numpy()
            if feats is None:
                feats=np.lib.format.open_memmap(outpath,mode="w+",dtype=np.float32,shape=(len(ids),z.shape[1]))
            feats[cursor:cursor+len(z)]=z; cursor+=len(z)
            del x,z
    feats.flush(); del feats
    return np.load(outpath,mmap_mode="r",allow_pickle=False),time.perf_counter()-started

def run_one(name,device,train_x,val_x,test_x,ytrain,yval,ytest,batch,epochs):
    backbone,architecture=make_backbone(name,device)
    n=len(train_x)
    arrays={
      "train":(train_x,np.arange(len(train_x))),
      "val":(val_x,np.arange(len(val_x))),
      "test":(test_x,np.arange(len(test_x)))
    }
    start=time.perf_counter()
    f={}
    for split,(arr,ids) in arrays.items():
        fp=OUT/f"{name}_{split}_pooled_features.npy"
        f[split],elapsed=extract_features(backbone,name,arr,ids,299 if name=="inception_v3" else 224,batch,device,fp)
        print(f"  cached {split} features: {f[split].shape}; extract {elapsed:.1f}s",flush=True)
    total_extract=time.perf_counter()-start
    d=f["train"].shape[1]
    head=nn.Linear(d,2).to(device)
    optim=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss()
    tx=torch.from_numpy(np.array(f["train"],copy=True)); ty=torch.from_numpy(ytrain.astype(np.int64,copy=True))
    vx=torch.from_numpy(np.array(f["val"],copy=True)); vy=torch.from_numpy(yval.astype(np.int64,copy=True))
    loader=DataLoader(TensorDataset(tx,ty),batch_size=512,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    best=float("inf"); best_state=None; stale=0; history={"loss":[],"val_loss":[],"accuracy":[],"val_accuracy":[]}
    head_start=time.perf_counter()
    for epoch in range(epochs):
        head.train(); losses=[]; correct=total=0
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device); optim.zero_grad(set_to_none=True)
            logits=head(xb); loss=loss_fn(logits,yb); loss.backward(); optim.step()
            losses.append(loss.item()); correct+=(logits.argmax(1)==yb).sum().item(); total+=len(yb)
        head.eval()
        with torch.no_grad():
            v_logits=head(vx.to(device)); vl=loss_fn(v_logits,vy.to(device)).item(); vp=v_logits.argmax(1).cpu().numpy()
        history["loss"].append(float(np.mean(losses))); history["accuracy"].append(correct/total)
        history["val_loss"].append(vl); history["val_accuracy"].append(float(accuracy_score(yval,vp)))
        print(f"  {name} epoch {epoch+1}/{epochs}: loss={np.mean(losses):.4f} val_loss={vl:.4f} val_acc={history['val_accuracy'][-1]:.4f}",flush=True)
        if vl<best:
            best=vl; best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=5: break
    head_seconds=time.perf_counter()-head_start
    head.load_state_dict(best_state); head.eval()
    torch.save({"head":head.state_dict(),"feature_dim":d,"backbone":architecture,"pretrained":True,"class_names":CLASS_NAMES},OUT/f"{name}_head.pt")
    with torch.no_grad():
        vpred=head(torch.from_numpy(np.array(f["val"],copy=True)).to(device)).argmax(1).cpu().numpy()
        t0=time.perf_counter(); tpred=head(torch.from_numpy(np.array(f["test"],copy=True)).to(device)).argmax(1).cpu().numpy(); infer=time.perf_counter()-t0
    test_report=classification_report(ytest,tpred,labels=[0,1],target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    cm=confusion_matrix(ytest,tpred,labels=[0,1]).tolist()
    result={"model":name,"features":"ImageNet-pretrained frozen backbone, global average pooled","architecture_or_weights":architecture,
      "hyperparameters":{"seed":SEED,"batch_size_feature_extraction":batch,"head":"Linear(AdamW lr=0.001 weight_decay=0.0001)","epochs_run":len(history["loss"]),"early_stopping_patience":5,"checkpoint":"lowest validation loss","input_size":299 if name=="inception_v3" else 224},
      "validation_accuracy":float(accuracy_score(yval,vpred)),"test_accuracy":float(accuracy_score(ytest,tpred)),
      "test_precision_damaged":test_report["Damaged"]["precision"],"test_recall_damaged":test_report["Damaged"]["recall"],"test_f1_damaged":test_report["Damaged"]["f1-score"],
      "test_confusion_matrix_labels_0_damaged_1_undamaged":cm,"train_seconds_including_feature_extraction":total_extract+head_seconds,
      "feature_extraction_seconds":total_extract,"head_fit_seconds":head_seconds,"inference_ms_per_image_head_only":1000*infer/len(ytest),
      "result_type":"local experiment","caveat":"Frozen feature extraction; this is transfer learning of a linear head, not historical end-to-end training."}
    (OUT/f"{name}_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    (OUT/f"{name}_test_report.json").write_text(json.dumps(classification_report(ytest,tpred,labels=[0,1],target_names=CLASS_NAMES,output_dict=True,zero_division=0),indent=2),encoding="utf-8")
    (OUT/f"{name}_test_predictions.npy").write_bytes(b"") if False else None
    np.save(OUT/f"{name}_test_predictions.npy",tpred)
    fig,ax=plt.subplots(figsize=(4,3.5)); ax.imshow(cm,cmap="Blues"); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,xlabel="Predicted",ylabel="Actual",title=f"{name} test confusion matrix")
    for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha="center",va="center")
    fig.tight_layout(); fig.savefig(OUT/f"{name}_confusion_matrix.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(history["loss"],label="train loss"); ax.plot(history["val_loss"],label="validation loss"); ax.legend(); ax.set(xlabel="Epoch",ylabel="Cross entropy",title=f"{name} linear head"); fig.tight_layout(); fig.savefig(OUT/f"{name}_learning_curve.png",dpi=160); plt.close(fig)
    (OUT/f"{name}_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"{name}: val={result['validation_accuracy']:.4f}, test={result['test_accuracy']:.4f}, total={result['train_seconds_including_feature_extraction']:.1f}s",flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",nargs="+",default=["mobilenet_v1","inception_v3","resnet50"],choices=["mobilenet_v1","inception_v3","resnet50"])
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--epochs",type=int,default=30)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU not detected; CPU deep-model run is disabled due expected runtime. Use the classical stage or install a CUDA-enabled PyTorch build.")
    device=torch.device("cuda")
    print("GPU:",torch.cuda.get_device_name(0),"memory GiB:",round(torch.cuda.get_device_properties(0).total_memory/1024**3,2),flush=True)
    Xtr=np.load(DATA/"task2_X_train.npy",mmap_mode="r",allow_pickle=False); Ytr=np.load(DATA/"task2_y_train.npy",mmap_mode="r",allow_pickle=False)
    Xte=np.load(DATA/"task2_X_test.npy",mmap_mode="r",allow_pickle=False); Yte=np.load(DATA/"task2_y_test.npy",mmap_mode="r",allow_pickle=False)
    labels=Ytr.argmax(1).astype(np.int64); ytest=Yte.argmax(1).astype(np.int64)
    tr,va=train_test_split(np.arange(len(labels)),test_size=.1,random_state=SEED,stratify=labels)
    results=[]
    for name in args.models:
        torch.cuda.empty_cache()
        results.append(run_one(name,device,Xtr[tr],Xtr[va],Xte,labels[tr],labels[va],ytest,args.batch_size,args.epochs))
    old=[]
    p=OUT/"deep_results.json"
    if p.exists(): old=json.loads(p.read_text(encoding="utf-8"))
    for item in results:
        old=[x for x in old if x["model"]!=item["model"]]; old.append(item)
    p.write_text(json.dumps(old,indent=2),encoding="utf-8")
    with (OUT/"deep_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["model","validation_accuracy","test_accuracy","test_precision_damaged","test_recall_damaged","test_f1_damaged","train_seconds_including_feature_extraction","feature_extraction_seconds","head_fit_seconds","inference_ms_per_image_head_only"]); writer.writeheader(); writer.writerows([{k:r[k] for k in writer.fieldnames} for r in old])

if __name__=="__main__": main()
