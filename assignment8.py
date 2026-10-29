#!/usr/bin/env python3
"""Assignment 8: controlled comparison of all four MiniSom neighborhoods.

Adapted independently from the experimental setup specified in the assigned guide:
https://www.datacamp.com/tutorial/self-organizing-maps
Run: python assignment8.py --output-dir .
"""
import argparse
import csv
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

FUNCTIONS=["gaussian","bubble","mexican_hat","triangle"]
CONFIG={"grid":[8,8],"sigma":1.5,"learning_rate":.5,
        "decay_function":"linear_decay_to_zero",
        "sigma_decay_function":"linear_decay_to_one",
        "activation_distance":"euclidean","topology":"rectangular",
        "iterations":5000,"primary_seed":123}


def fit(data,func,seed):
    # MiniSom warns (by design) that noninteger sigma is unusual for bubble/triangle.
    # Preserve 1.5 in every run to change ONLY the assigned neighborhood function.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        som=MiniSom(8,8,data.shape[1],sigma=1.5,learning_rate=.5,
                    decay_function="linear_decay_to_zero",sigma_decay_function="linear_decay_to_one",
                    neighborhood_function=func,activation_distance="euclidean",
                    topology="rectangular",random_seed=seed)
        som.pca_weights_init(data)
    som.train_random(data,5000,verbose=False)
    return som


def diagnostics(som,data,target):
    winners=np.array([som.winner(row) for row in data])
    weights=som.get_weights()
    # hypot avoids overflow when the Mexican-hat repulsive ring sends unused
    # prototypes far from the data under the tutorial's fixed learning rate.
    stable_distances=np.hypot.reduce(data[:,None,None,:]-weights[None,:,:,:],axis=-1)
    stable_bmu=np.argmin(stable_distances.reshape(len(data),-1),axis=1)
    assert np.array_equal(stable_bmu,winners[:,0]*8+winners[:,1])
    cell=winners[:,0]*8+winners[:,1]
    occupied=np.unique(cell)
    mixed=sum(len(np.unique(target[cell==v]))>1 for v in occupied)
    correct=sum(np.bincount(target[cell==v],minlength=3).max() for v in occupied)
    # Verify library QE against its definition (mean Euclidean distance, NOT MSE).
    qe=np.mean(np.linalg.norm(data-som.get_weights()[winners[:,0],winners[:,1]],axis=1))
    assert np.isclose(qe,som.quantization_error(data),rtol=1e-12,atol=1e-12)
    return {"quantization_error":float(qe),"topographic_error":float(som.topographic_error(data)),
            "occupied_cells":len(occupied),"mixed_cells":int(mixed),
            "cell_majority_purity":float(correct/len(data)),
            "max_abs_weight":float(np.max(np.abs(weights)))},winners


def stable_u_matrix(som):
    """MiniSom's rectangular, normalized neighbor-sum U matrix without overflow.

    Dividing ALL weights by a common positive scale cancels on normalization.
    The trained weights and the BMU assignments are left untouched.
    """
    weights=som.get_weights()
    scaled=weights/max(1.,float(np.max(np.abs(weights))))
    u=np.zeros((8,8))
    for x in range(8):
        for y in range(8):
            for dx,dy in [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]:
                if 0<=x+dx<8 and 0<=y+dy<8:
                    u[x,y]+=np.linalg.norm(scaled[x,y]-scaled[x+dx,y+dy])
    result=u/u.max() if u.max()>0 else u
    assert np.isfinite(result).all()
    if np.max(np.abs(weights))<1e100:
        assert np.allclose(result,som.distance_map(),rtol=1e-12,atol=1e-12)
    return result


def panel(ax,som,winners,target,jitter,title):
    ax.pcolormesh(np.arange(9),np.arange(9),stable_u_matrix(som).T,
                  cmap="Greys",vmin=0,vmax=1,alpha=.30,shading="flat",rasterized=True)
    palette=["#0072B2","#D55E00","#009E73"]
    markers=["o","^","s"]
    for c,name in enumerate(["Setosa","Versicolor","Virginica"]):
        mask=target==c
        # Jitter is only for display and cannot move a point out of its BMU cell.
        ax.scatter(winners[mask,0]+.5+jitter[mask,0],winners[mask,1]+.5+jitter[mask,1],
                   color=palette[c],marker=markers[c],s=23,alpha=.85,edgecolor="white",lw=.25,label=name)
    ax.set(xlim=(0,8),ylim=(0,8),xticks=np.arange(9),yticks=np.arange(9),
           xlabel="SOM x cell",ylabel="SOM y cell",title=title,aspect="equal")
    ax.grid(alpha=.25,lw=.6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    for sub in ["figures","results","models"]:(out/sub).mkdir(exist_ok=True)
    iris=load_iris();data=MinMaxScaler().fit_transform(iris.data);target=iris.target
    np.savez_compressed(out/"results"/"iris_data.npz",raw=iris.data,normalized=data,target=target)
    # Same seed, PCA initialization, training sample order and display offsets for every function.
    jitter=np.random.default_rng(456).uniform(-.4,.4,size=(len(data),2))
    primary=[];models={};winners_all={};repeats=[]
    for func in FUNCTIONS:
        for seed in [123,124,125,126,127]:
            som=fit(data,func,seed);row,winners=diagnostics(som,data,target)
            row={"neighborhood":func,"seed":seed,**row};repeats.append(row)
            if seed==123:
                primary.append(row);models[func]=som;winners_all[func]=winners
                np.savez_compressed(out/"models"/f"{func}.npz",weights=som.get_weights(),winners=winners)
            print(json.dumps(row),flush=True)
    with (out/"results"/"primary_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(primary[0]));w.writeheader();w.writerows(primary)
    with (out/"results"/"repeated_seed_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(repeats[0]));w.writeheader();w.writerows(repeats)
    summary=[]
    for func in FUNCTIONS:
        q=[r["quantization_error"] for r in repeats if r["neighborhood"]==func]
        summary.append({"neighborhood":func,"mean_qe":float(np.mean(q)),"sd_qe":float(np.std(q,ddof=1)),
                        "min_qe":min(q),"max_qe":max(q)})
    result={"config":CONFIG,"primary":primary,"repeats_summary":summary,
            "versions":{n:version(n) for n in ["MiniSom","numpy","scikit-learn","matplotlib"]},
            "python":platform.python_version(),"notes":[
                "Labels are used only for visualization and post-training diagnostics, never for training.",
                "MinMaxScaler is fit on all Iris points, matching the unsupervised descriptive tutorial; this is not held-out prediction accuracy.",
                "MiniSom warns for noninteger sigma in bubble/triangle. Sigma=1.5 is deliberately kept fixed to isolate neighborhood choice.",
                "Class indices are 0,1,2, mapped directly to their correct names and colors; the tutorial's c-1 indexing is corrected.",
                "PCA initialization is identical for all functions; random seed affects the random training sample order.",
                "Topographic error uses MiniSom's rectangular-map convention (diagonal adjacency included).",
                "Mexican-hat prototypes diverge under the fixed tutorial hyperparameters; their magnitude is recorded. No retuning or clipping is applied.",
                "The normalized U matrix is computed after common rescaling of all weights to prevent overflow, leaving its mathematical definition unchanged.",
                "All BMUs are cross-checked using overflow-safe hypot distances."
            ]}
    (out/"results"/"all_results.json").write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout="constrained")
    for ax,row in zip(axes.flat,primary):
        func=row["neighborhood"]
        panel(ax,models[func],winners_all[func],target,jitter,
              f"{func.replace('_',' ').title()} | QE={row['quantization_error']:.5f}")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3)
    fig.suptitle("Iris SOM: data points within BMU cells (seed 123)",fontsize=14)
    fig.savefig(out/"figures"/"fig3_neighborhood_comparison.png",dpi=180)
    plt.close(fig)
    for row in primary:
        func=row["neighborhood"];fig,ax=plt.subplots(figsize=(6,5.5),layout="constrained")
        panel(ax,models[func],winners_all[func],target,jitter,
              f"{func.replace('_',' ').title()} | QE={row['quantization_error']:.5f}")
        ax.legend(fontsize=8,loc="upper right")
        fig.savefig(out/"figures"/f"fig3_{func}.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.7),layout="constrained")
    pos=np.arange(4)
    ax.bar(pos,[s["mean_qe"] for s in summary],yerr=[s["sd_qe"] for s in summary],
           color=["#0072B2","#E69F00","#D55E00","#009E73"],capsize=5,alpha=.8)
    for j,func in enumerate(FUNCTIONS):
        q=[r["quantization_error"] for r in repeats if r["neighborhood"]==func]
        ax.scatter(j+np.linspace(-.12,.12,5),q,color="black",s=20,zorder=5)
    ax.set(xticks=pos,xticklabels=[f.replace('_',' ').title() for f in FUNCTIONS],
           ylabel="Quantization error",title="Five common seeds (123-127): mean +/- sample SD")
    ax.grid(axis="y",alpha=.25)
    fig.savefig(out/"figures"/"quantization_error_comparison.png",dpi=180);plt.close(fig)
    print("DONE",flush=True)


if __name__=="__main__":main()
