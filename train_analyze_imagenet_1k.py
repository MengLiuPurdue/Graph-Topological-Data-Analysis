"""
This file creates two Reeb networks to compare ResNet50 v.s. AlexNet and VOLO v.s. ResNet50 on the ImageNet-1k dataset.
This file assumes the ImageNet-1k training and validation dataset is organized as:
dataset/imagenet_1k/
    train/
        class1/
            img1
            img2
            ...
        class2/
        ...
    val/
        class1/
            img1
            img2
            ...
        class2/
        ...
This file only creates the two reeb networks, one for ResNet50 v.s. AlexNet and the other for VOLO v.s. ResNet50. 
The precomputed graphs are not included due to large file size and can be provided upon request.
For more details on embedding images on any reeb net component of interest, see 'analyze_imagenette.ipynb' for a demo.
"""
#%%
from tqdm import tqdm
from GTDA.models import GenericCNNData
from GTDA.GTDA_utils import compute_reeb, NN_model
from GTDA.GTDA import GTDA
from torchvision import transforms
from argparse import Namespace
import torchvision.models as torch_models
from GTDA.GTDA_utils import SPoC,normalize,knn_cuda_graph
from sklearn.decomposition import PCA
import torch
import numpy as np
import scipy.sparse as sp
import os
import os.path as osp
import torch.nn.functional as F
import timm
#%%
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose(
    [
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(img_mean, img_std),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(img_mean, img_std),
    ]
)
#%%
args = {
    "batch_size": 128,
    "data_root": "/scratch2/liu1740/ImageNet",
    "num_workers": 8,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
}
args = Namespace(**args)
#%%
args.shuffle = False
args.drop_last = False
data_orig = GenericCNNData(
    args,train_transform=test_transform,
    test_transform=test_transform,train_subdir="train",
    test_subdir="val")
trainset_orig = data_orig.train_dataset
testset_orig = data_orig.test_dataset
trainloader_orig = data_orig.train_dataloader()
testloader_orig = data_orig.test_dataloader()
#%%
cnn_model_resnet = torch_models.resnet50(pretrained=True).cuda()
cnn_model_resnet.eval()
_,y,preds_resnet = SPoC(cnn_model_resnet,[trainloader_orig,testloader_orig],pooling='avg')
y = np.array(y)
X_resnet,_,_ = SPoC(cnn_model_resnet,[trainloader_orig,testloader_orig],pooling='max')
pca_resnet = PCA(n_components=128,random_state=42)
Xr_resnet = pca_resnet.fit_transform(X_resnet)
Dinv = sp.spdiags(1/pca_resnet.singular_values_,0,Xr_resnet.shape[1],Xr_resnet.shape[1])
Xr_resnet = Xr_resnet@Dinv
Xr_resnet = normalize(Xr_resnet)
Xr_resnet = torch.tensor(Xr_resnet).to('cuda')
A_knn_resnet = knn_cuda_graph(Xr_resnet,5,256)
A_knn_resnet = (A_knn_resnet>0).astype(np.float64)
Xr_resnet.cpu()

cnn_model_alexnet = torch_models.alexnet(pretrained=True).cuda()
cnn_model_alexnet.eval()
preds_alexnet = []
for dataloader in [trainloader_orig,testloader_orig]:
    cnt = len(dataloader)
    for inputs,labels in tqdm(dataloader,total=cnt):
        with torch.set_grad_enabled(False):
            tmp = cnn_model_alexnet(inputs.cuda())
            preds_alexnet.append(F.softmax(tmp,dim=1).cpu().detach().numpy())
preds_alexnet = np.concatenate(preds_alexnet)

volo_model = timm.create_model('volo_d5_224', pretrained=True)
volo_model.eval()
volo_model = volo_model.cuda()
X_volo = []
y = []
preds_volo = []
with torch.no_grad():
    for dataloader in [trainloader_orig,testloader_orig]:
        cnt = len(dataloader)
        for inputs,labels in tqdm(dataloader,total=cnt):
            inputs = inputs.cuda()
            batch_feats = volo_model.forward_features(inputs)
            batch_feats = batch_feats.mean(dim=1)
            batch_preds = volo_model(inputs)
            X_volo.append(batch_feats.cpu().detach().numpy())
            preds_volo.append(F.softmax(batch_preds,dim=1).cpu().detach().numpy())

X_volo = np.concatenate(X_volo)
preds_volo = np.concatenate(preds_volo)
pca_volo = PCA(n_components=128,random_state=42)
Xr_volo = pca_volo.fit_transform(X_volo)

Dinv = sp.spdiags(1/pca_volo.singular_values_,0,Xr_volo.shape[1],Xr_volo.shape[1])
Xr_volo = Xr_volo@Dinv
Xr_volo = normalize(Xr_volo)
Xr_volo = torch.tensor(Xr_volo).to('cuda')
A_knn_volo = knn_cuda_graph(Xr_volo,5,256)
A_knn_volo = (A_knn_volo>0).astype(np.float64)
Xr_volo.cpu()
#%%
"""
Build Reeb network to compare ResNet50 v.s. AlexNet
"""
ntrains = len(data_orig.train_dataset.imgs)
nn_model = NN_model()
nn_model.preds = preds_resnet
nn_model.labels = labels
nn_model.A = A_knn_resnet
nn_model.train_mask = np.zeros(A_knn_resnet.shape[0])
nn_model.train_mask[0:ntrains] = True
nn_model.val_mask = np.zeros(A_knn_resnet.shape[0])
nn_model.test_mask = np.zeros(A_knn_resnet.shape[0])
nn_model.test_mask[ntrains::] = True
smallest_component = 25
overlap = (0,0.01) # only extend left bin's right boundary by 10%
labels_to_eval = list(range(preds_resnet.shape[1]))
GTDA_record_1 = compute_reeb(GTDA,nn_model,labels_to_eval,smallest_component,overlap,extra_lens=preds_alexnet,
    node_size_thd=5,reeb_component_thd=5,nprocs=10,device='cuda',split_thd=0.001,nsteps_preprocess=10)
#%%
"""
Build Reeb network to compare VOLO v.s. ResNet50
"""
ntrains = len(data_orig.train_dataset.imgs)
nn_model = NN_model()
nn_model.preds = preds_volo
nn_model.labels = labels
nn_model.A = A_knn_volo
nn_model.train_mask = np.zeros(A_knn_volo.shape[0])
nn_model.train_mask[0:ntrains] = True
nn_model.val_mask = np.zeros(A_knn_volo.shape[0])
nn_model.test_mask = np.zeros(A_knn_volo.shape[0])
nn_model.test_mask[ntrains::] = True
smallest_component = 25
overlap = 0.01
labels_to_eval = list(range(preds_volo.shape[1]))
GTDA_record_2 = compute_reeb(GTDA,nn_model,labels_to_eval,smallest_component,overlap,extra_lens=preds_resnet,
    node_size_thd=5,reeb_component_thd=5,nprocs=10,device='cuda',split_thd=0.001,nsteps_preprocess=10)