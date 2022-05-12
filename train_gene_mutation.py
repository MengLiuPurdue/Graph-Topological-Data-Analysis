"""
This file creates the gene variants KNN graph and trains a logistic regression model to predict harmful v.s. non-harmful 
gene variants from variants embeddings.
'mutation_dataset.py' needs to be run first to extract embeddings from a pretrained Enformer model.
See 'dataset/precomputed/variants' for precomputed graph and lens to reproduce figures from the paper.
The graphs and lens might be different among different runs due to randomness of model training. 
"""
#%%
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from GTDA.GTDA_utils import normalize, knn_cuda_graph
import torch
import scipy.sparse as sp
import os
import os.path as osp
# %%
root = "dataset"
dataset = "variants"
selected_gene = "BRCA1"
model_save_dir = f"{root}/{dataset}/{selected_gene}"

all_ref_preds = np.load(f"{model_save_dir}/reference_predictions.npy")
all_alt_preds = np.load(f"{model_save_dir}/alternative_predictions.npy")
labels = np.load(f"{model_save_dir}/labels.npy")

targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')
cage_features = [i for i in range(df_targets.shape[0]) if 'CAGE' in df_targets.iloc[i]['description']]
for preds in [all_ref_preds,all_alt_preds]:
    for i in tqdm(cage_features,total=len(cage_features)):
        preds[:,:,i] = np.log(1+preds[:,:,i])

preds_diff = []
for i in range(all_ref_preds.shape[0]):
    preds_diff.append(
        (np.sum(all_ref_preds[i],0)-np.sum(all_alt_preds[i],0)).tolist())
preds_diff = np.array(preds_diff)
ss = StandardScaler()
preds_diff = ss.fit_transform(preds_diff)
#%%
np.random.seed(42)
train_ids = np.random.choice(range(preds_diff.shape[0]),int(0.5*preds_diff.shape[0]))
test_ids = np.setdiff1d(range(preds_diff.shape[0]),train_ids)
clf = LogisticRegression(penalty='l1',solver='liblinear')
clf.fit(preds_diff[train_ids,:],labels[train_ids])
preds = clf.predict_proba(preds_diff)
#%%
pca = PCA(n_components=128,random_state=42)
Xr = pca.fit_transform(preds_diff)
Dinv = sp.spdiags(1/pca.singular_values_,0,Xr.shape[1],Xr.shape[1])
Xr = Xr@Dinv
Xr = normalize(Xr)
knn = 5
A_knn = knn_cuda_graph(
    torch.tensor(Xr).cuda(),knn,256)
extra_lens=Xr[:,0:2]
#%%
A_knn = A_knn.tocoo()
savepath = f"{root}/precomputed/{dataset}"
if not osp.isdir(savepath):
    os.makedirs(savepath)
with open(f"{savepath}/edge_list.txt","w") as f:
    f.write(f"{A_knn.shape[0]} {A_knn.shape[0]} {A_knn.nnz}\n")
    for ei,ej in zip(A_knn.row,A_knn.col):
        f.write(f"{ei} {ej} 1\n")
np.save(f"{savepath}/prediction_lens.npy",preds)
np.save(f"{savepath}/labels.npy",labels)
np.save(f"{savepath}/extra_lens.npy",extra_lens)
with open(f"{savepath}/train_nodes.txt","w") as f:
    for node in train_ids:
        f.write(f"{node}\n")
with open(f"{savepath}/test_nodes.txt","w") as f:
    for node in test_ids:
        f.write(f"{node}\n")