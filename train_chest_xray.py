"""
This file trains a DenseNet-121 model to predict diseases from chest X-ray images.
We use the CheXNet implementation from 'https://github.com/zoogzog/chexnet', which should be downloaded and put under the same directory of this file.
This file assumes the chest X-ray images are put under 'dataset/chest_xray/images' and expert labels under 'dataset/chest_xray'.
Such files can be obtained from 'https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest'.
See 'dataset/precomputed/chest_xray' for precomputed graph and lens to reproduce figures from the paper.
Your results might be different from precomputed ones due to randomness of model training as well as hardware difference. 
"""
#%%
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
from GTDA.GTDA_utils import *
from GTDA.GTDA import GTDA
import shutil
from sklearn.decomposition import PCA
from tqdm import tqdm 
import time
# %%
import sys
sys.path.append('chexnet')
from DensenetModels import *
from DatasetGenerator import DatasetGenerator
from ChexnetTrainer import ChexnetTrainer
# %%
DENSENET121 = 'DENSE-NET-121'

timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime

#---- Path to the directory with images
root = "dataset"
dataset = "chest_xray"
pathDirData = f"{root}/{dataset}"
#%%
test_list = pd.read_table(f'{pathDirData}/test_list.txt',delimiter=' ',names=['name'])
train_val_list = pd.read_table(f'{pathDirData}/train_val_list.txt',delimiter=' ',names=['name'])
df_labels = pd.read_table(f'{pathDirData}/Data_Entry_2017_v2020.csv',delimiter=',')
# %%
test_ids = test_list['name'].values
patient_ids = np.sort(np.unique([img_id.split('_')[0] for img_id in train_val_list['name'].values]))
# %%
np.random.seed(42)
val_patient_ids = set(np.random.choice(patient_ids,int(0.2*len(patient_ids))))
train_patient_ids = set(np.setdiff1d(patient_ids,val_patient_ids))
test_patient_ids = set([i.split('_')[0] for i in test_ids])
# %%
all_findings = set(np.concatenate([i.split('|') for i in df_labels['Finding Labels'].values]))
all_findings.remove('No Finding')
all_findings = sorted(list(all_findings))

train_strs = []
val_strs = []
test_strs = []
for i in tqdm(range(df_labels.shape[0])):
    cols = df_labels.iloc[i]
    img_index = cols['Image Index']
    findings = cols['Finding Labels']
    labels = []
    for finding in all_findings:
        labels.append(str(int(finding in findings)))
    if img_index.split('_')[0] in val_patient_ids:
        val_strs.append(' '.join(['images/'+img_index]+labels))
    elif img_index.split('_')[0] in test_patient_ids:
        test_strs.append(' '.join(['images/'+img_index]+labels))
    else:
        train_strs.append(' '.join(['images/'+img_index]+labels))
# %%
with open(f"{pathDirData}/train_labels.txt","w") as f:
    f.write("\n".join(train_strs))
with open(f"{pathDirData}/val_labels.txt","w") as f:
    f.write("\n".join(val_strs))
with open(f"{pathDirData}/test_labels.txt","w") as f:
    f.write("\n".join(test_strs))

#---- Paths to the files with training, validation and testing sets.
#---- Each file should contains pairs [path to image, output vector]
#---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
pathFileTrain = f'{pathDirData}/train_labels.txt'
pathFileVal = f'{pathDirData}/val_labels.txt'
pathFileTest = f'{pathDirData}/test_labels.txt'

#---- Neural network parameters: type of the network, is it pre-trained 
#---- on imagenet, number of classes
nnArchitecture = DENSENET121
nnIsTrained = True
nnClassCount = 14

#---- Training settings: batch size, maximum number of epochs
trBatchSize = 16
trMaxEpoch = 100

#---- Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = 256
imgtransCrop = 224
    
pathModel = 'm-' + timestampLaunch + '.pth.tar'
# pathModel = 'm-12042022-190659.pth.tar'
print ('Training NN architecture = ', nnArchitecture)
ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
#%%
"""
Load trained model and perform GTDA analysis
"""
model = DenseNet121(14,True)
model = torch.nn.DataParallel(model).cuda() 
modelCheckpoint = torch.load(f"results/chest_xray/m-{timestampLaunch}.pth.tar")
model.load_state_dict(modelCheckpoint['state_dict'])
model.eval()
model = list(model.children())[0]
model = list(model.children())[0]
# %%
feature_extractor = model.features
classifier = model.classifier
feature_extractor.eval()
classifier.eval()
# %%
img_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transResize = 256
transCrop = 224
trBatchSize = 16

transformList = []
transformList.append(transforms.Resize(transResize))
transformList.append(transforms.TenCrop(transCrop))
transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
transformList.append(transforms.Lambda(lambda crops: torch.stack([img_normalize(crop) for crop in crops])))
transformSequence=transforms.Compose(transformList)

#-------------------- SETTINGS: DATASET BUILDERS
datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
        
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=False,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)

datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
#%%
t1 = time.time()
outGT = torch.FloatTensor().cuda()
outPRED = torch.FloatTensor().cuda()
outFEATS = torch.FloatTensor()
for dataLoader in [dataLoaderTrain,dataLoaderVal,dataLoaderTest]:
    n = len(dataLoader)
    for i, (input, target) in tqdm(enumerate(dataLoader),total=n):
        target = target.cuda()
        outGT = torch.cat((outGT, target), 0)
        
        bs, n_crops, c, h, w = input.size()
        with torch.no_grad():
            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
            feats = feature_extractor(varInput)
            feats = F.relu(feats, inplace=True)
            feats_max = torch.clone(feats)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = torch.flatten(feats, 1)
            out = classifier(feats)
            feats_max = F.adaptive_max_pool2d(feats_max, (1, 1))
            feats_max = torch.flatten(feats_max, 1)
        featsMean = feats_max.view(bs, n_crops, -1).mean(1).cpu().detach()
        outMean = out.view(bs, n_crops, -1).mean(1)
        outPRED = torch.cat((outPRED, outMean.data), 0)
        outFEATS = torch.cat((outFEATS, featsMean.data), 0)
t2 = time.time()
print(f"Predicting and embedding took: {t2-t1} seconds.")
#%%
aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, 14)
aurocMean = np.array(aurocIndividual).mean()

print ('AUROC mean ', aurocMean)

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

for i in range (0, len(aurocIndividual)):
    print (CLASS_NAMES[i], ' ', aurocIndividual[i])
#%%
# Use training and validation data to find threshold that can maximize F1 score
ntrains = len(datasetTrain)+len(datasetVal)
preds = np.max(outPRED.cpu().detach().numpy(),1)[0:ntrains]
labels = (np.sum(outGT.cpu().detach().numpy(),1)>0).astype(np.int64)[0:ntrains]
best_F1 = -1*float('inf')
best_thd = -1
true_pos = 0
true_neg = len(labels)-np.sum(labels)
for i,j in enumerate(np.argsort(-1*preds)):
    true_pos += labels[j]
    true_neg -= (1-labels[j])
    pr = true_pos/(i+1)
    rc = true_pos/np.sum(labels)
    f1 = 2*pr*rc/(pr+rc) if pr+rc > 0 else 0
    if f1 > best_F1:
        best_thd = preds[j]
        best_F1 = f1
print(best_F1)
preds = np.max(outPRED.cpu().detach().numpy(),1)/best_thd*0.5
preds[np.nonzero(preds>1)[0]] = 1
#%%
pca = PCA(n_components=64,random_state=42)
Xr = pca.fit_transform(outFEATS.numpy())
Dinv = sp.spdiags(1/pca.singular_values_,0,Xr.shape[1],Xr.shape[1])
Xr = Xr@Dinv
Xr = normalize(Xr)
extra_lens = Xr[:,0:2]
Xr = torch.tensor(Xr).to('cuda')
A_knn = knn_cuda_graph(Xr,5,256,thd=0)
A_knn = (A_knn>0).astype(np.float64)
n = A_knn.shape[0]
nn_model = NN_model()
nn_model.preds = preds.reshape((-1,1))
nn_model.preds = np.hstack([1-nn_model.preds,nn_model.preds])
nn_model.A = A_knn
nn_model.labels = (np.sum(outGT.cpu().detach().numpy(),1)>0).astype(np.int64)
nn_model.train_mask = np.zeros(n,dtype=np.bool_)
nn_model.train_mask[0:ntrains] = True
nn_model.val_mask = np.zeros(n,dtype=np.bool_)
nn_model.test_mask = np.zeros(n,dtype=np.bool_)
nn_model.test_mask[ntrains::] = True

labels_to_eval = [0,1]
extra_lens = outPRED.cpu().detach().numpy()
smallest_component = 50
overlap = 0.005
GTDA_record = compute_reeb(GTDA,nn_model,labels_to_eval,smallest_component,overlap,extra_lens=None,
    node_size_thd=5,reeb_component_thd=5,nprocs=10,device='cuda',nsteps_preprocess=10)
# %%
savepath = "dataset/precomputed/chest_xray"
A_knn = A_knn.tocoo()
train_ids = np.nonzero(GTDA_record['nn_model'].train_mask)[0]
test_ids = np.nonzero(GTDA_record['nn_model'].test_mask)[0]
all_test_img_names = [i.split('/')[-1] for i in datasetTest.listImagePaths]

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
with open(f"{savepath}/test_img_names.txt","w") as f:
    for name in all_test_img_names:
        f.write(f"{name}\n")
