import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def TheDotDotPlot(X,Y,Z,xlabel,data,filename):
    plt.rcParams['font.family'] = 'Lato'
    fig,ax = plt.subplots(figsize=(16,10))
    im = plt.scatter(X,Y,c=Z,cmap='viridis')
    cb = plt.colorbar()
    cb.set_label('UTI Diagnoisis',fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)
    plt.errorbar(data.index,"mean",yerr="std",data=data,fmt=" ",c="#FF0000")
    plt.scatter(data.index,"mean",data=data,marker="_",c="#FF0000")
    n=len(data.index)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(["Negative","Other","Small","Moderate","Large"])
    #plt.locator_params(axis="x", integer=True, tight=True)
    ax.tick_params(axis='y',labelsize=10)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel("SHAP values",fontsize=15)
    fig.savefig(filename+".eps",bbox_inches="tight")

df = pd.read_csv("dataset1.txt")
df1 = pd.read_csv("dataset1_shap.txt")
Z = np.loadtxt("result.txt")
i=0
var=["ua_wbc"]
varlen=len(var)

for i in range(varlen):
    X=df[var[i]]
    #dX = (np.random.rand(len(X))-0.5)*0.75
    #X=X+dX
    print(X.shape)
    Y=df1[var[i]+'_shap']
    print(Y.shape)
    dff = pd.concat([X, Y], axis=1)
    dff.to_csv("aaa.txt")
    dff=pd.read_csv("aaa.txt")
    X=dff[var[i]]
    Y=dff[var[i]+'_shap']
    dX = (np.random.rand(len(X))-0.5)*0.75
    X = X+dX
    x_label=var[i]
    group=dff.groupby(var[i])[var[i]+"_shap"].agg([np.mean,np.std])
    print(group)
    #plt.scatter(X,Y,c=Z,cmap='viridis')
    TheDotDotPlot(X,Y,Z,x_label,group,var[i])

    #plt.scatter(X,Y)
plt.show()
