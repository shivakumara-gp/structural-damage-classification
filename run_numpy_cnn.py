"""A small CNN trained with NumPy only (no SciPy, PyTorch or compiled ML DLLs).

This is a local fallback experiment, not MobileNetV1/InceptionV3/ResNet50.
Inputs are the already-created 32x32 RGB cache from run_local.py.
"""
from __future__ import annotations
import json, time
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"; OUT=ROOT/"artifacts"/"local"
SEED=42; NAMES=["Damaged","Undamaged"]

def split_ids(y):
    rng=np.random.default_rng(SEED); train=[]; val=[]
    for cls in (0,1):
        ids=np.flatnonzero(y==cls); rng.shuffle(ids); nval=round(.1*len(ids))
        val.extend(ids[:nval]); train.extend(ids[nval:])
    return np.asarray(train),np.asarray(val)

def conv_forward(x,w,b,stride):
    # NHWC im2col convolution. sliding_window_view only creates a view.
    k=w.shape[0]
    windows=np.lib.stride_tricks.sliding_window_view(x,(k,k),axis=(1,2))[:,::stride,::stride]
    cols=windows.transpose(0,1,2,4,5,3).reshape(-1,k*k*x.shape[-1])
    out=(cols@w.reshape(-1,w.shape[-1])+b).reshape(x.shape[0],windows.shape[1],windows.shape[2],-1)
    return out,(cols,x.shape,stride,w)

def conv_backward(dout,cache):
    cols,xshape,stride,w=cache; flat=dout.reshape(-1,dout.shape[-1]);
    dw=(cols.T@flat).reshape(w.shape); db=flat.sum(axis=0)
    dcols=flat@w.reshape(-1,w.shape[-1]).T
    b,h,width,c=xshape; k=w.shape[0]; oh=dout.shape[1]; ow=dout.shape[2]
    dwindows=dcols.reshape(b,oh,ow,k,k,c); dx=np.zeros(xshape,dtype=np.float32)
    for iy in range(k):
        for ix in range(k):
            dx[:,iy:iy+oh*stride:stride,ix:ix+ow*stride:stride,:]+=dwindows[:,:,:,iy,ix,:]
    return dx,dw,db

def forward(x,p,cache=False):
    z1,c1=conv_forward(x,p["w1"],p["b1"],2); a1=np.maximum(z1,0)
    z2,c2=conv_forward(a1,p["w2"],p["b2"],2); a2=np.maximum(z2,0)
    pooled=a2.mean(axis=(1,2)); logits=pooled@p["wf"]+p["bf"]
    if cache: return logits,(z1,c1,z2,c2,a2,pooled)
    return logits

def backward(dlogits,cache,p):
    z1,c1,z2,c2,a2,pooled=cache; b=len(dlogits)
    grads={"wf":pooled.T@dlogits,"bf":dlogits.sum(0)}
    dpool=dlogits@p["wf"].T
    da2=np.broadcast_to(dpool[:,None,None,:]/(a2.shape[1]*a2.shape[2]),a2.shape).copy()
    dz2=da2*(z2>0); da1,dw2,db2=conv_backward(dz2,c2)
    dz1=da1*(z1>0); _,dw1,db1=conv_backward(dz1,c1)
    grads.update(w1=dw1,b1=db1,w2=dw2,b2=db2)
    return grads

def softmax_loss(logits,y):
    shifted=logits-logits.max(axis=1,keepdims=True); e=np.exp(shifted); probs=e/e.sum(axis=1,keepdims=True)
    loss=-np.log(np.maximum(probs[np.arange(len(y)),y],1e-12)).mean()
    probs[np.arange(len(y)),y]-=1; probs/=len(y)
    return float(loss),probs

def predict_batches(X,p,batch=64):
    result=[]
    for i in range(0,len(X),batch): result.append(forward(np.asarray(X[i:i+batch],dtype=np.float32),p).argmax(1))
    return np.concatenate(result)

def score(y,pred):
    cm=[[int(((y==i)&(pred==j)).sum()) for j in (0,1)] for i in (0,1)]
    classes={}
    for c,name in enumerate(NAMES):
        tp=cm[c][c]; fp=cm[1-c][c]; fn=cm[c][1-c]
        pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
        classes[name]={"precision":pr,"recall":re,"f1-score":2*pr*re/(pr+re) if pr+re else 0,"support":int((y==c).sum())}
    return {"accuracy":float((y==pred).mean()),"confusion_matrix_labels_0_damaged_1_undamaged":cm,"classification_report":classes}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    feature_file=OUT/"X_train_32_rgb.npy"
    if not feature_file.exists(): raise SystemExit("Create feature caches first: python run_local.py --stage classical")
    X=np.load(feature_file,mmap_mode="r",allow_pickle=False)
    y=np.load(DATA/"task2_y_train.npy",mmap_mode="r").argmax(1).astype(np.int64)
    Xt=np.load(OUT/"X_test_32_rgb.npy",mmap_mode="r",allow_pickle=False)
    yt=np.load(DATA/"task2_y_test.npy",mmap_mode="r").argmax(1).astype(np.int64)
    tri,vi=split_ids(y); rng=np.random.default_rng(SEED)
    # The fixed paper scaling yields [-1,1]; neural input uses RGB [0,1].
    p={"w1":rng.normal(0,.08,(3,3,3,8)).astype(np.float32),"b1":np.zeros(8,np.float32),
       "w2":rng.normal(0,.06,(3,3,8,16)).astype(np.float32),"b2":np.zeros(16,np.float32),
       "wf":rng.normal(0,.08,(16,2)).astype(np.float32),"bf":np.zeros(2,np.float32)}
    m={k:np.zeros_like(v) for k,v in p.items()}; v={k:np.zeros_like(a) for k,a in p.items()}
    lr=.003; b1=.9; b2=.999; eps=1e-8; step=0; history={"train_loss":[],"val_loss":[],"train_accuracy":[],"val_accuracy":[]}
    best=float("inf"); bestp=None; stale=0; start_time=time.perf_counter(); max_epochs=12; batch=32
    for epoch in range(1,max_epochs+1):
        order=tri.copy(); rng.shuffle(order); losses=[]; correct=total=0
        for start in range(0,len(order),batch):
            ids=order[start:start+batch]; xb=np.asarray(X[ids],dtype=np.float32)*.5+.5; yb=y[ids]
            logits,cache=forward(xb,p,True); loss,dl=softmax_loss(logits,yb); grads=backward(dl,cache,p)
            for k in p:
                g=grads[k]; m[k]=b1*m[k]+(1-b1)*g; v[k]=b2*v[k]+(1-b2)*g*g
                mh=m[k]/(1-b1**step) if step else m[k]; vh=v[k]/(1-b2**step) if step else v[k]
                # Adam step counter advances below; use the next-step correction.
                mh=m[k]/(1-b1**(step+1)); vh=v[k]/(1-b2**(step+1))
                p[k]-=lr*mh/(np.sqrt(vh)+eps)
            step+=1; losses.append(loss); correct+=(logits.argmax(1)==yb).sum(); total+=len(yb)
        vx=np.asarray(X[vi],dtype=np.float32)*.5+.5; vy=y[vi]
        vl,vdl=softmax_loss(forward(vx,p),vy); vpred=forward(vx,p).argmax(1)
        train_loss=float(np.mean(losses)); train_acc=float(correct/total); val_acc=float((vpred==vy).mean())
        history["train_loss"].append(train_loss); history["val_loss"].append(vl); history["train_accuracy"].append(train_acc); history["val_accuracy"].append(val_acc)
        print(f"NumpyTinyCNN epoch {epoch}/{max_epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_loss={vl:.4f}",flush=True)
        if vl<best-1e-4: best=vl; bestp={k:a.copy() for k,a in p.items()}; stale=0
        else:
            stale+=1
            if stale>=3: break
    train_s=time.perf_counter()-start_time; p=bestp
    np.savez(OUT/"NumpyTinyCNN.npz",**p)
    vp=predict_batches(np.asarray(X[vi],dtype=np.float32)*.5+.5,p); tp0=time.perf_counter(); tp=predict_batches(np.asarray(Xt,dtype=np.float32)*.5+.5,p); infer_s=time.perf_counter()-tp0
    val=score(y[vi],vp); test=score(yt,tp)
    result={"model":"NumpyTinyCNN (alternative)","architecture":"32x32 RGB; Conv3x3 stride2 3->8, Conv3x3 stride2 8->16, global average pool, dense 16->2; trained from scratch with NumPy/Adam",
      "hyperparameters":{"seed":SEED,"optimizer":"Adam","learning_rate":lr,"batch_size":batch,"max_epochs":max_epochs,"epochs_run":len(history["train_loss"]),"early_stopping_patience":3,"best_validation_loss":best},
      "validation_accuracy":val["accuracy"],"test_accuracy":test["accuracy"],"test_precision_damaged":test["classification_report"]["Damaged"]["precision"],
      "test_recall_damaged":test["classification_report"]["Damaged"]["recall"],"test_f1_damaged":test["classification_report"]["Damaged"]["f1-score"],
      "test_confusion_matrix_labels_0_damaged_1_undamaged":test["confusion_matrix_labels_0_damaged_1_undamaged"],"train_seconds":train_s,"inference_ms_per_image_model_only":1000*infer_s/len(yt),
      "result_type":"local experiment","limitation":"Small CNN alternative; does not reproduce MobileNetV1, InceptionV3, ResNet50, or pretrained transfer learning."}
    (OUT/"NumpyTinyCNN_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (OUT/"NumpyTinyCNN_reports.json").write_text(json.dumps({"validation":val,"test":test},indent=2),encoding="utf-8")
    np.save(OUT/"NumpyTinyCNN_test_predictions.npy",tp)
    (OUT/"NumpyTinyCNN_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    for kind,ids in (("correct",np.flatnonzero(tp==yt)),("wrong",np.flatnonzero(tp!=yt))):
        chosen=ids[:8]; fig,axs=plt.subplots(1,len(chosen),figsize=(2.5*len(chosen),3)); axs=np.atleast_1d(axs)
        for ax,i in zip(axs,chosen):
            rgb=np.clip((np.asarray(Xt[int(i)])+1)*.5,0,1); ax.imshow(rgb); ax.set_title(f"true={NAMES[yt[i]]}\npred={NAMES[tp[i]]}",fontsize=8); ax.axis("off")
        fig.tight_layout(); fig.savefig(OUT/f"NumpyTinyCNN_test_{kind}_examples.png",dpi=130); plt.close(fig)
    cm=test["confusion_matrix_labels_0_damaged_1_undamaged"]
    fig,ax=plt.subplots(figsize=(4,3.5)); ax.imshow(cm,cmap="Blues"); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=NAMES,yticklabels=NAMES,xlabel="Predicted",ylabel="Actual",title="NumPy Tiny CNN test confusion matrix")
    for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha="center",va="center")
    fig.tight_layout(); fig.savefig(OUT/"NumpyTinyCNN_confusion_matrix.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(history["train_loss"],label="train"); ax.plot(history["val_loss"],label="validation"); ax.set(xlabel="Epoch",ylabel="Cross entropy",title="NumPy CNN learning curve"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"NumpyTinyCNN_learning_curve.png",dpi=160); plt.close(fig)
    classical=json.loads((OUT/"classical_results.json").read_text(encoding="utf-8"))
    allrows=classical+[result]
    (OUT/"all_model_results.json").write_text(json.dumps(allrows,indent=2),encoding="utf-8")
    with (OUT/"all_model_results.csv").open("w",newline="",encoding="utf-8") as f:
        cols=["model","validation_accuracy","test_accuracy","test_precision_damaged","test_recall_damaged","test_f1_damaged","train_seconds","inference_ms_per_image_model_only"]
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows([{k:r.get(k) for k in cols} for r in allrows])
    fig,ax=plt.subplots(figsize=(9,4.5)); names=[r["model"] for r in allrows]; pos=np.arange(len(names)); width=.38
    ax.bar(pos-width/2,[r["validation_accuracy"] for r in allrows],width,label="Validation")
    ax.bar(pos+width/2,[r["test_accuracy"] for r in allrows],width,label="Test")
    ax.set(xticks=pos,xticklabels=names,ylim=(0,1),ylabel="Accuracy",title="Measured local model comparison"); ax.tick_params(axis="x",rotation=18); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"all_model_accuracy_comparison.png",dpi=160); plt.close(fig)
    print(f"NumpyTinyCNN: val={val['accuracy']:.4f}, test={test['accuracy']:.4f}, fit={train_s:.1f}s",flush=True)

if __name__=="__main__": main()
