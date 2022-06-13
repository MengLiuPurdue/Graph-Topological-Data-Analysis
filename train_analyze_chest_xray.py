"""
This file trains a DenseNet-121 model to predict diseases from chest X-ray images and then use GTDA to find 
which test image labels might be wrong and compare with expert labels.
This file assumes the chest X-ray images are put under 'dataset/chest_xray/images' and expert labels under 'dataset/chest_xray'.
Such files can be obtained from 'https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest'.
The results might be different among different runs due to randomness of model training. 
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
#%%
# with open('../understand_GNN/results/chest_xray/GTDA_record.p','rb') as f:
#     GTDA_record = pickle.load(f)
# nn_model = GTDA_record['nn_model']
# n = nn_model.preds.shape[0]
# ntrains = np.sum(nn_model.train_mask)+np.sum(nn_model.val_mask)
pred_labels = np.argmax(nn_model.preds,1)
gtda = GTDA_record['gtda']
g_reeb = GTDA_record['g_reeb']
reeb_components = find_components(g_reeb,size_thd=0)[1]
reeb_components_to_nodes = {}
for i,reeb_component in enumerate(reeb_components):
    nodes = []
    for reeb_node in reeb_component:
        nodes += gtda.final_components_filtered[gtda.filtered_nodes[reeb_node]]
    if len(nodes) > 0:
        reeb_components_to_nodes[i] = np.unique(nodes)

node_to_component_id = np.array([-1]*n)
for i,component in reeb_components_to_nodes.items():
    for node in component:
        node_to_component_id[node] = i
#%%
expert_labels = pd.read_table(f'{pathDirData}/all_findings_expert_labels/test_labels.csv',delimiter=',')
expert_tested_imgs = expert_labels['Image ID'].values
expert_labels = expert_labels['Abnormal'].values
expert_labels[np.nonzero(expert_labels=='NO')] = 0
expert_labels[np.nonzero(expert_labels=='YES')] = 1
expert_labels = expert_labels.astype(np.int64)
test_img_ids = {}
for i,img in enumerate(datasetTest.listImagePaths):
    test_img_ids[img.split('/')[-1]] = ntrains+i

expert_tested_imgs = np.array([test_img_ids[i] for i in expert_tested_imgs])
expert_tested_incorrect_imgs = expert_tested_imgs[np.nonzero(nn_model.labels[expert_tested_imgs] != expert_labels)[0]]
print("Type\tExpert_Labels_in_Component\tIncorrect_by_Experts\tFlagged_as_Problematic\tPrecision\tRecall")
#%%
thd = 0.5
all_estimate = []
all_labels = []
cnt = Counter([node_to_component_id[k] for k in expert_tested_incorrect_imgs])
cnt_experts = Counter([node_to_component_id[k] for k in expert_tested_imgs])
# Single component
component_indices = []
all_tp = 0
all_num_total = 0
all_num_pos = 0
all_num_experts = 0
for i,k in cnt.items(): 
    if k >= 3:
        component_indices.append(i)
for component_index,num_pos in cnt.most_common(len(component_indices)):
    tp = 0
    num_total = 0
    num_experts = 0
    for i in reeb_components_to_nodes[component_index]:
        if i in set(expert_tested_imgs):
            num_experts += 1
            error = int(nn_model.labels[i]!=pred_labels[i])
            num_total += (np.abs(error-gtda.sample_colors_mixing[i])>thd)
            tp += ((i in set(expert_tested_incorrect_imgs)) * (np.abs(error-gtda.sample_colors_mixing[i])>thd))
            all_estimate.append(np.abs(error-gtda.sample_colors_mixing[i]))
            all_labels.append(int(i in set(expert_tested_incorrect_imgs)))
    all_tp += tp
    all_num_total += num_total
    all_num_pos += num_pos
    all_num_experts += num_experts
    print(f"Single_Component\t{num_experts}\t{num_pos}\t{num_total}\t{round(tp/num_total,2)}\t{round(tp/num_pos,2)}")
# Components with 2 incorrect expert labels
component_indices = []
for i,k in cnt.items(): 
    if k == 2:
        component_indices.append(i)
tp = 0
num_total = 0
num_pos = 0
num_experts = 0
for component_index in component_indices:
    num_pos += cnt[component_index]
    for i in reeb_components_to_nodes[component_index]:
        if i in set(expert_tested_imgs):
            num_experts += 1
            error = int(nn_model.labels[i]!=pred_labels[i])
            num_total += (np.abs(error-gtda.sample_colors_mixing[i])>thd)
            tp += ((i in set(expert_tested_incorrect_imgs)) * (np.abs(error-gtda.sample_colors_mixing[i])>thd))
            all_estimate.append(np.abs(error-gtda.sample_colors_mixing[i]))
            all_labels.append(int(i in set(expert_tested_incorrect_imgs)))
all_tp += tp
all_num_total += num_total
all_num_pos += num_pos
all_num_experts += num_experts
print(f"Components_with_2_incorrect_labels\t{num_experts}\t{num_pos}\t{num_total}\t{round(tp/num_total,2)}\t{round(tp/num_pos,2)}")
# Components with 1 incorrect expert label
component_indices = []
for i,k in cnt.items(): 
    if k == 1:
        component_indices.append(i)
tp = 0
num_total = 0
num_pos = 0
num_experts = 0
for component_index in component_indices:
    num_pos += cnt[component_index]
    for i in reeb_components_to_nodes[component_index]:
        if i in set(expert_tested_imgs):
            num_experts += 1
            error = int(nn_model.labels[i]!=pred_labels[i])
            num_total += (np.abs(error-gtda.sample_colors_mixing[i])>thd)
            tp += ((i in set(expert_tested_incorrect_imgs)) * (np.abs(error-gtda.sample_colors_mixing[i])>thd))
            all_estimate.append(np.abs(error-gtda.sample_colors_mixing[i]))
            all_labels.append(int(i in set(expert_tested_incorrect_imgs)))
all_tp += tp
all_num_total += num_total
all_num_pos += num_pos
all_num_experts += num_experts
print(f"Components_with_1_incorrect_label\t{num_experts}\t{num_pos}\t{num_total}\t{round(tp/num_total,2)}\t{round(tp/num_pos,2)}")
# Components with 0 incorrect expert label
component_indices = []
for i,k in cnt_experts.items(): 
    if i not in cnt:
        component_indices.append(i)
tp = 0
num_total = 0
num_pos = 0
num_experts = 0
for component_index in component_indices:
    num_pos += cnt[component_index]
    for i in reeb_components_to_nodes[component_index]:
        if i in set(expert_tested_imgs):
            num_experts += 1
            error = int(nn_model.labels[i]!=pred_labels[i])
            num_total += (np.abs(error-gtda.sample_colors_mixing[i])>thd)
            tp += ((i in set(expert_tested_incorrect_imgs)) * (np.abs(error-gtda.sample_colors_mixing[i])>thd))
            all_estimate.append(np.abs(error-gtda.sample_colors_mixing[i]))
            all_labels.append(int(i in set(expert_tested_incorrect_imgs)))
all_tp += tp
all_num_total += num_total
all_num_pos += num_pos
all_num_experts += num_experts
print(f"Components_with_0_incorrect_label\t{num_experts}\t{num_pos}\t{num_total}\t{round(tp/num_total,2)}\tNaN")
# Summarize and save to a tex file
print(f"Overall\t{all_num_experts}\t{all_num_pos}\t{all_num_total}\t{round(all_tp/all_num_total,2)}\t{round(all_tp/all_num_pos,2)}")
# %%