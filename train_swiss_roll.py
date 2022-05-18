"""
This is a self-contained file that creates a synthetic 3-class Swiss Roll dataset and train a 2-layer GCN model to predict node labels. 
See 'dataset/precomputed/swiss_roll' for precomputed graph and lens to reproduce figures from the paper. 
The graphs and lens might be different among different runs due to randomness of model training. 
"""
#%%
from GTDA.models import GCN
from GTDA.GTDA_utils import *
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
import shutil
import copy
from argparse import Namespace
from sklearn.decomposition import PCA
# %%
n_samples = 1000
noise = 1.2
X,y = make_swiss_roll(n_samples=n_samples,noise=noise,random_state=123)
X = np.vstack([X[:,0],X[:,2]]).T
labels = np.array([0]*len(y))
labels[np.nonzero(y<=np.quantile(y,0.33))] = 0
labels[np.nonzero((y>np.quantile(y,0.33))*(y<=np.quantile(y,0.66)))] = 1
labels[np.nonzero(y>np.quantile(y,0.66))] = 2
G = kneighbors_graph(X, n_neighbors=5, include_self=False)
G = ((G+G.T)>0).astype(np.int)
# %%
name = "swiss_roll"
root = "dataset"
savepath = f"{root}/{name}"
if os.path.isdir(savepath):
    shutil.rmtree(savepath)
dataset = data_generator(G,X,labels,name,root_path=root)
data = dataset[0]
# %%
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
            model.to(device)

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
    best_model.to(device)
    return test_acc, best_val_acc, best_model, data
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
torch.manual_seed(123)
model = GCN(data.x.shape[1],args.hidden,dataset.num_classes,2,args.dropout)
test_acc, best_val_acc, model, data = run(args, dataset, data, model)
print(f'test acc = {test_acc:.4f} \t val acc = {best_val_acc:.4f}')
# %%
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
pca = PCA(n_components=16,random_state=0)
Xr = pca.fit_transform(node_embs)
Dinv = sp.spdiags(1/pca.singular_values_,0,Xr.shape[1],Xr.shape[1])
Xr = Xr@Dinv
Xr = normalize(Xr)
knn = 2
A_knn = knn_cuda_graph(
    torch.tensor(Xr).cuda(),knn,256)
# augmenting the original graph with a KNN from embeddings
A_all = ((A_knn+A)>0).astype(np.float32)
# %%
A = A.tocoo()
A_all = A_all.tocoo()
savepath = f"{root}/precomputed/{name}"
if not osp.isdir(savepath):
    os.makedirs(savepath)
with open(f"{savepath}/edge_list_orig.txt","w") as f:
    f.write(f"{A.shape[0]} {A.shape[0]} {A.nnz}\n")
    for ei,ej in zip(A.row,A.col):
        f.write(f"{ei} {ej} 1\n")
with open(f"{savepath}/edge_list.txt","w") as f:
    f.write(f"{A_all.shape[0]} {A_all.shape[0]} {A_all.nnz}\n")
    for ei,ej in zip(A_all.row,A_all.col):
        f.write(f"{ei} {ej} 1\n")
np.save(f"{savepath}/prediction_lens.npy",preds)
np.save(f"{savepath}/labels.npy",labels)
np.save(f"{savepath}/coords.npy",X)
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
# %%
