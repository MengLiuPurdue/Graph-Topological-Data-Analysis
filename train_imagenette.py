"""
This file retrains the last layer of a pretrained ResNet50 model on 10 easy classes of ImageNet.
Images need to be downloaded from https://github.com/fastai/imagenette and put under 'dataset/imagenette' folder.
See 'dataset/precomputed/imagenette' for precomputed graph and lens to reproduce figures from the paper.
The graphs and lens might be different among different runs due to randomness of model training. 
"""
#%%
import glob
from os.path import isfile
import shutil
from tqdm import tqdm
from GTDA.models import ResNetModule,GenericCNNData
from torchvision import transforms
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace
import torchvision.models as torch_models
from GTDA.GTDA_utils import SPoC,normalize,knn_cuda_graph
from sklearn.decomposition import PCA
import torch
import numpy as np
import scipy.sparse as sp
import os
import os.path as osp
#%%
root = "dataset"
dataset = "imagenette"
all_files = glob.glob(f"{root}/{dataset}/**",recursive=True)
for name in tqdm(all_files,total=len(all_files)):
    if isfile(name):
        if 'val/' in name and 'val_' not in name:
            shutil.move(name,name.replace('val/','train/'))
        if 'train/' in name and 'val_' in name:
            shutil.move(name,name.replace('train/','val/'))
# %%
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
    "data_root": f"{root}/{dataset}",
    "weight_decay": 1e-2,
    "learning_rate": 1e-2,
    "max_epochs": 5,
    "num_workers": 8,
    "num_classes": 10,
    "precision": 32,
    "gpu_id": 0,
    "pretrained": True,
    "download": True,
    "shuffle": True,
    "drop_last": False,
    "pin_memory": True,
    "lr_warmup": 0.2,
    "lr_gamma": 0.1,
    "lr_step_size": 30,
    "crop_size": 256,
    "emb_crop_size": 256,
}
args = Namespace(**args)

checkpoint = ModelCheckpoint(
    monitor="acc/val", mode="max", save_last=True)
seed_everything(42, workers=True)
lightning_model = ResNetModule(args)
logger = TensorBoardLogger("Imagenette", name="resnet50")
trainer = Trainer(
    logger=logger,
    gpus=-1,
    deterministic=True,
    weights_summary=None,
    log_every_n_steps=1,
    max_epochs=args.max_epochs,
    callbacks=[checkpoint],
    precision=args.precision,
)
data = GenericCNNData(
    args,train_transform=train_transform,
    test_transform=test_transform,train_subdir="train",
    test_subdir="val")
assert(data.train_dataset.class_to_idx==data.test_dataset.class_to_idx)
#%%
trainer.fit(lightning_model,data)
# %%
model = torch_models.resnet50(pretrained=True)
model.fc = lightning_model.classifier
cnn_model = model.to('cuda')
cnn_model.eval()
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
_,y,preds_orig = SPoC(cnn_model,[trainloader_orig,testloader_orig],pooling='avg')
y = np.array(y)
X_orig,_,_ = SPoC(cnn_model,[trainloader_orig,testloader_orig],pooling='max')
# %%
pca = PCA(n_components=128,random_state=42)
Xr_orig = pca.fit_transform(X_orig)
Dinv = sp.spdiags(1/pca.singular_values_,0,Xr_orig.shape[1],Xr_orig.shape[1])
Xr_orig = Xr_orig@Dinv
Xr_orig = normalize(Xr_orig)
Xr_orig = torch.tensor(Xr_orig).to('cuda')
A_knn_orig = knn_cuda_graph(Xr_orig,5,256)
A_knn = (A_knn_orig>0).astype(np.float64)
# %%
A_knn = A_knn.tocoo()
savepath = f"{root}/precomputed/{dataset}"
if not osp.isdir(savepath):
    os.makedirs(savepath)
with open(f"{savepath}/edge_list.txt","w") as f:
    f.write(f"{A_knn.shape[0]} {A_knn.shape[0]} {A_knn.nnz}\n")
    for ei,ej in zip(A_knn.row,A_knn.col):
        f.write(f"{ei} {ej} 1\n")
np.save(f"{savepath}/prediction_lens.npy",preds_orig)
np.save(f"{savepath}/labels.npy",y)
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
imgs_list = data_orig.train_dataset.imgs+data_orig.test_dataset.imgs
with open(f"{savepath}/imgs_list.txt","w") as f:
    for img,_ in imgs_list:
        f.write(f"{img}\n")