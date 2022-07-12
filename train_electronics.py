"""
This file creates the Amazon electronics graph from reviews data and trains a 2-layer GCN model to predict product categories.
All products under 'Electronics' from the 2014 version of Amazon reviews data from http://jmcauley.ucsd.edu/data/amazon/index_2014.html 
needs to be put under 'dataset' folder to run this file.
See 'dataset/precomputed/electronics' for precomputed graph and lens to reproduce figures from the paper.
Your results might be different from precomputed ones due to randomness of model training as well as hardware difference. 
"""
#%%
import pandas as pd
import gzip
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from argparse import Namespace
import torch
import torch.nn.functional as F
import copy
import os
import os.path as osp
import shutil
from sklearn.decomposition import PCA
from GTDA.GTDA_utils import *
from GTDA.models import GCN
#%%
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
      yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')
#%%
dataset = "Electronics_2014"
root = "dataset"
df = getDF(f'{root}/meta_{dataset}.json.gz')

label_to_name = {
    0:"Desktops",
    1:"Data Storage",
    2:"Laptops",
    3:"Monitors",
    4:"Computer Components",
    5:"Video Projectors",
    6:"Routers",
    7:"Tablets",
    8:"Networking Products",
    9:"Webcams"}
name_to_label = {v:k for k,v in label_to_name.items()}
selected_nodes = []
node_labels = []
node_id_to_asin = {}
node_asin_to_id = {}
for i in range(df.shape[0]):
    cats = df['categories'].values[i][0]
    for cat in cats:
        if cat in name_to_label:
            selected_nodes.append(i)
            node_labels.append(name_to_label[cat])
            node_id_to_asin[i] = df.iloc[i]['asin']
            node_asin_to_id[node_id_to_asin[i]] = i
            break

df_reviews = getDF(f'{root}/reviews_{dataset}.json.gz')
#%%
grouped = df_reviews.groupby('asin')
product_reviews = {}
for name,group in tqdm(grouped):
    if name in node_asin_to_id:
        product_reviews[name] = " ".join(group['reviewText'].values)

valid_actions = ['also_bought', 'bought_together', 'buy_after_viewing']
ei = []
ej = []
selected_nodes_set = set(selected_nodes)
for node_id in selected_nodes:
    product = df.iloc[node_id]
    actions = product['related']
    if type(actions) == dict:
        for action in valid_actions:
            if action in actions:
                neighs = actions[action]
                for neigh in neighs:
                    if neigh in node_asin_to_id and node_asin_to_id[neigh] in selected_nodes_set:
                        ei.append(node_id)
                        ej.append(node_asin_to_id[neigh])
ndim = np.max(selected_nodes)+1
A = sp.csr_matrix((np.ones(len(ei)),(ei,ej)),(ndim,ndim))
A = ((A+A.T)>0).astype(np.int64)
Ar = A[selected_nodes,:][:,selected_nodes]
filtered_nodes, _ = find_components(Ar,size_thd=100)
Ar = Ar[filtered_nodes,:][:,filtered_nodes]
#%%
product_reviews_list = []
for i in filtered_nodes:
    asin = node_id_to_asin[selected_nodes[i]]
    if asin in product_reviews:
        product_reviews_list.append(product_reviews[asin])
    else:
        product_reviews_list.append("")
product_reviews_list = np.array(product_reviews_list)
count_vector = TfidfVectorizer(max_df=0.5,min_df=0.01)
count_vector.fit(product_reviews_list)
doc_arrays = []
for i in tqdm(range(0,len(product_reviews_list),10000)):
    doc_array = sp.csr_matrix(count_vector.transform(product_reviews_list[i:min(i+10000,len(product_reviews_list))]).toarray())
    doc_arrays.append(doc_array)

doc_array = sp.vstack(doc_arrays)
vmap = {v:k for k,v in count_vector.vocabulary_.items()}
labels = node_labels[filtered_nodes]
#%%
def run(args, dataset, data, model):

    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[data.train_mask]
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data.x, data.edge_index), [], [], []
        logits = F.log_softmax(logits, dim=1)
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data.x, data.edge_index)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = random_splits(
        data, dataset.num_classes, train_percent=args.train_rate,
        val_percent=args.val_rate,seed=123)
    model, data = model.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_model = None
    for epoch in range(args.epochs):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_model = copy.deepcopy(model.cpu())
            model.to('cuda:0')

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.min().item():
                    break
    if best_model is None:
        best_model = copy.deepcopy(model.cpu())
    best_model.to('cuda:0')
    return test_acc, best_val_acc, best_model, data
#%%
name = "electronics"
root = "dataset"
savepath = f"{root}/{name}"
if os.path.isdir(savepath):
    shutil.rmtree(savepath)
dataset = data_generator(Ar,doc_array,labels,name,root_path=root)
data = dataset[0]
#%%
args = {
    "train_rate": 0.1,
    "val_rate": 0.1,
    "lr": 0.01,
    "weight_decay": 1.0e-5,
    "hidden": 64,
    "epochs": 500,
    "early_stopping": 10,
    "dropout": 0.5,
}
args = Namespace(**args)
dname = args.dataset

train_rate = args.train_rate
val_rate = args.val_rate
percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
val_lb = int(round(val_rate*len(data.y)))
TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
print('True Label rate: ', TrueLBrate)

torch.manual_seed(123)
model = GCN(data.x.shape[1],args.hidden,dataset.num_classes,2,args.dropout)
test_acc, best_val_acc, model, data = run(args, dataset, data, model)

print(f'test acc = {test_acc:.4f} \t val acc = {best_val_acc:.4f}')
#%%
from GTDA.GTDA import *
from GTDA.GTDA_utils import *

model.eval()
with torch.set_grad_enabled(False):
    preds = F.softmax(model(data.x, data.edge_index))
preds = preds.cpu().detach().numpy()
with torch.set_grad_enabled(False):
    node_embs = F.relu(model.convs[0](data.x, data.edge_index))
node_embs = node_embs.cpu().detach().numpy()
n = preds.shape[0]
edge_index = data.edge_index.cpu().detach().numpy()
A = sp.csr_matrix((np.ones(edge_index.shape[1]),(edge_index[0],edge_index[1])),shape=(n,n))
A = ((A+A.T)>0).astype(np.float64)

pca = PCA(n_components=16,random_state=123)
Xr = pca.fit_transform(node_embs)
Dinv = sp.spdiags(1/pca.singular_values_,0,Xr.shape[1],Xr.shape[1])
Xr = Xr@Dinv
Xr = normalize(Xr)
knn = 2
A_knn = knn_cuda_graph(
    torch.tensor(Xr).cuda(),knn,256)
# augmenting the original graph with a KNN from embeddings
A_all = ((A_knn+A)>0).astype(np.float32)
#%%
A_all = A_all.tocoo()
savepath = f"{root}/precomputed/{name}"
if not osp.isdir(savepath):
    os.makedirs(savepath)
with open(f"{savepath}/edge_list.txt","w") as f:
    f.write(f"{A_all.shape[0]} {A_all.shape[0]} {A_all.nnz}\n")
    for ei,ej in zip(A_all.row,A_all.col):
        f.write(f"{ei} {ej} 1\n")
np.save(f"{savepath}/prediction_lens.npy",preds)
np.save(f"{savepath}/labels.npy",labels)
train_nodes = np.nonzero(data.train_mask.cpu().detach().numpy())[0]
val_nodes = np.nonzero(data.val_mask.cpu().detach().numpy())[0]
test_nodes = np.nonzero(data.test_mask.cpu().detach().numpy())[0]
with open(f"{savepath}/train_nodes.txt","w") as f:
    for node in train_nodes:
        f.write(f"{node}\n")
with open(f"{savepath}/val_nodes.txt","w") as f:
    for node in val_nodes:
        f.write(f"{node}\n")
with open(f"{savepath}/test_nodes.txt","w") as f:
    for node in test_nodes:
        f.write(f"{node}\n")
all_asin = df['asin'].values
with open(f"{savepath}/product_asin.txt","w") as f:
    for node in filtered_nodes:
        f.write(f"{all_asin[selected_nodes[node]]}\n")