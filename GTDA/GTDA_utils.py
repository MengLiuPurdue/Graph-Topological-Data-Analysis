import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import numpy as np
from collections import Counter
import torch
from torch_geometric.data import InMemoryDataset
from torch_cluster import knn_graph
from torch_geometric.data import Data
from datetime import datetime
import pickle
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import normalize
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import greedypacker
from matplotlib.lines import Line2D
import pandas as pd
import torch.nn as nn
import gc
import os
import os.path as osp
import math
import time


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    '''
    if file name not specified, use time stamp.
    '''
    now = datetime.now()
    surfix = now.strftime('%b_%d_%Y-%H:%M')
    if file_name is None:
        tmp_data_name = '_'.join(['cSBM_data', surfix])
    else:
        tmp_data_name = file_name
    p2cSBM_data = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2cSBM_data, 'bw') as f:
        pickle.dump(data, f)
    return p2cSBM_data

def random_splits(data, num_classes, train_percent=0.1, val_percent=0.1, seed=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing
    generator = torch.Generator()
    indices = []
    train_sizes = []
    val_sizes = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        generator.manual_seed(seed*1000000+i)
        index = index[torch.randperm(index.size(0),generator=generator)]
        indices.append(index)
        train_sizes.append(int(round(len(index)*train_percent)))
        val_sizes.append(int(round(len(index)*val_percent)))

    train_index = torch.cat(
        [index[0:train_sizes[i]] for i,index in enumerate(indices)], dim=0)
    val_index = torch.cat(
        [index[train_sizes[i]:(val_sizes[i]+train_sizes[i])] for i,index in enumerate(indices)], dim=0)
    test_index = torch.cat(
        [index[(val_sizes[i]+train_sizes[i]):] for i,index in enumerate(indices)], dim=0)

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(
        test_index, size=data.num_nodes)
    return data

def compute_cond(A,cluster):
    degs = np.asarray(np.sum(A,0))[0]
    vol = np.sum(degs[cluster])
    cut = vol-np.sum(A[cluster,:][:,cluster])
    return cut/min(vol,np.sum(degs)-vol)


def filter_components(A,size_thd=100,verbose=False):
    if A.nnz == 0 and size_thd == 0:
        return list(range(A.shape[0])), [[i] for i in range(A.shape[0])]
    ret = sp.csgraph.connected_components(A,directed=False)
    if verbose:
        print(f"component sizes: {Counter(ret[1])}")
    components = [[] for i in range(ret[0])]
    for i,l in enumerate(ret[1]):
        components[l].append(i)
    selected_components = [list(c) for c in components if len(c)>size_thd]
    selected_nodes = []
    for c in selected_components:
        selected_nodes += c
    return selected_nodes, selected_components

def find_largest_component(A):
    components = filter_components(A,size_thd=0)[1]
    largest_component_id = np.argmax([len(c) for c in components])
    return components[largest_component_id]

def compute_coords(G,ncls,labels,niters=10):
    nnodes = G.shape[0]
    G1 = G.copy().tocoo()
    npts = Counter(labels)
    cls_center_loc = [(10*np.cos((i*1.0/ncls)*(2*np.pi)-(54/180)*(np.pi)),10*np.sin((i*1.0/ncls)*(2*np.pi)-(54/180)*(np.pi))) for i in range(ncls)]
    init_coords = []
    for k in range(ncls):
        npts_curr = npts[k]
        init_coords += [(cls_center_loc[k][0]+2*np.cos((i*1.0/npts_curr)*(2*np.pi)),cls_center_loc[k][1]+2*np.sin((i*1.0/npts_curr)*(2*np.pi))) for i in range(npts_curr)]
    init_coords = {i:init_coords[i] for i in range(nnodes)}
    nxG = nx.Graph()
    nxG.add_nodes_from(range(nnodes))
    nxG.add_edges_from(zip(G1.row,G1.col))
    coords = nx.spring_layout(
        nxG,pos=init_coords,iterations=niters,seed=0)
    coords = np.array([coords[i] for i in range(nnodes)])
    return coords


def build_knn_graph(node_embs,k,cosine=False):
    n = node_embs.shape[0]
    edge_index = knn_graph(
        node_embs,k=k,loop=False,cosine=cosine).cpu().numpy()
    knn = sp.csr_matrix(
        (np.ones(edge_index.shape[1]),
        (edge_index[0],edge_index[1])),
        shape=(n,n))
    return knn


class NN_model(object):
    def __init__(self,modelname):
        self.modelname = modelname
        self.preds = None
        self.model = None
        self.A = None
        self.labels = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

# convert any graph with feature matrix to InMemoryDataset
class data_generator(InMemoryDataset):
    def __init__(self,G,X,labels,name,root_path="data/",
        train_percent=0.1,val_percent=0.1,transform=None, pre_transform=None):
        self.name = name
        self.edge_index = np.vstack((
            G.tocoo().row.astype(np.int64),
            G.tocoo().col.astype(np.int64)))
        self.x = X
        self.y = labels
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.transform = transform
        self.pre_transform = pre_transform
        path = osp.join(root_path, self.name)
        if not osp.isdir(path):
            os.makedirs(path)
        super(data_generator, self).__init__(
            path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.raw_dir, name)
            num_class = len(np.unique(self.y))
            n = len(self.y)
            data = Data(x=torch.tensor(self.x, dtype=torch.float32),
                edge_index=torch.tensor(self.edge_index),
                y=torch.tensor(self.y, dtype=torch.int64))
            # order edge list and remove duplicates if any.
            data.coalesce()
            data = random_splits(
                data, num_class, self.train_percent, self.val_percent)
            data.num_class = num_class
            data.train_percent = self.train_percent
            data.val_percent = self.val_percent
            _ = save_data_to_pickle(data,
                                    p2root=self.raw_dir,
                                    file_name=self.name)

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        print(p2f)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

def compute_recall(labels,pre_labels,pred,test_only=True,test_nodes=None):
    if test_only:
        labels = labels[test_nodes]
        pre_labels = pre_labels[test_nodes]
        pred = pred[test_nodes]
    num_errors = np.sum(labels!=pre_labels)
    recalls = []
    curr = 0
    for i in np.argsort(-1*pred):
        curr += (pre_labels[i]!=labels[i])
        recalls.append(1.0*curr/num_errors)
    return recalls

def compute_recall_reeb(reeb_nodes,labels,pre_labels,pred,key_to_components):
    num_errors = np.sum(labels!=pre_labels)
    curr = 0
    recalls = [0]
    num_nodes = [0]
    curr_num = 0
    for i in np.argsort(-1*pred):
        key = reeb_nodes[i]
        component = key_to_components[key]
        curr += np.sum(pre_labels[component]!=labels[component])
        recalls.append(1.0*curr/num_errors)
        curr_num += len(component)
        num_nodes.append(curr_num)
    return recalls,num_nodes

def SPoC(model,dataloaders,key_map=None,is_normalize=True,indices_to_ignore=None,pooling="max"):
    X_all = defaultdict(list)
    y = []
    preds = []
    for dataloader in dataloaders:
        cnt = len(dataloader)
        for inputs,labels in tqdm(dataloader,total=cnt):
            with torch.set_grad_enabled(False):
                tmp = model.bn1(model.conv1(inputs.to('cuda')))
                tmp = model.relu(tmp)
                tmp = model.maxpool(tmp)
                tmp = model.layer1(tmp)
                tmp = model.layer2(tmp)
                X_all[2].append(tmp.flatten(start_dim=2).sum(dim=2).cpu().detach().numpy())
                tmp = model.layer3(tmp)
                X_all[1].append(tmp.flatten(start_dim=2).sum(dim=2).cpu().detach().numpy())
                tmp = model.layer4(tmp)
                if pooling == "max":
                    pooling_layer = torch.nn.AdaptiveMaxPool2d(output_size=(1,1)).cuda()
                    tmp = pooling_layer(tmp)
                else:
                    tmp = model.avgpool(tmp)
                X_all[0].append(tmp.flatten(start_dim=2).sum(dim=2).cpu().detach().numpy())
                y += labels.numpy().tolist()
                tmp = torch.flatten(tmp, 1)
                tmp = model.fc(tmp)
                if indices_to_ignore is not None:
                    tmp[:,indices_to_ignore] = -1*float('inf')
                preds.append(F.softmax(tmp,dim=1).cpu().detach().numpy())
    if is_normalize:
        X_all = {
            0:normalize(np.concatenate(X_all[0])),
            1:normalize(np.concatenate(X_all[1])),
            2:normalize(np.concatenate(X_all[2])),
        }
    else:
        X_all = {
            0:np.concatenate(X_all[0]),
            1:np.concatenate(X_all[1]),
            2:np.concatenate(X_all[2]),
        }
    preds = np.concatenate(preds)
    if key_map is not None:
        preds = preds[:,key_map]
    inputs.cpu()
    torch.cuda.empty_cache()
    return X_all,y,preds

def knn_cuda(features,train_features,k):
    similarity = torch.mm(features, train_features)
    distances, indices = similarity.topk(k, largest=True, sorted=True)
    return distances, indices

def knn_cuda_batched(features,train_features,k,batch_size,device='cuda'):
    features = features.to(device)
    overall_distances = None
    overall_indices = None
    for i in range(0,train_features.shape[0],batch_size):
        si = i
        ei = min(si+batch_size,train_features.shape[0])
        train_features_batch = train_features[si:ei,:].t().to(device)
        similarity = torch.mm(features, train_features_batch)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        distances = distances.detach().cpu()
        indices = indices.detach().cpu()+si
        if overall_distances is None:
            overall_distances = distances
            overall_indices = indices
        else:
            tmp_distances = torch.cat([overall_distances,distances],dim=1)
            tmp_indices = torch.cat([overall_indices,indices],dim=1)
            new_indices = torch.argsort(tmp_distances,dim=1,descending=True)[:,0:k]
            overall_distances = torch.gather(tmp_distances,1,new_indices)
            overall_indices = torch.gather(tmp_indices,1,new_indices)
    return overall_distances, overall_indices

def knn_cuda_graph(X,knn,batch_size,thd=0,device=None,batch_training=False,batch_size_training=50000):
    ei,ej = [],[]
    for i in tqdm(range(0,X.shape[0],batch_size)):
        start_i = i
        end_i = min(start_i+batch_size,X.shape[0])
        if batch_training:
            distances,batch_indices = knn_cuda_batched(X[start_i:end_i,:],X,knn+1,batch_size_training)
        else:
            distances,batch_indices = knn_cuda(X[start_i:end_i,:],X.t(),knn+1)
        for xi in range(batch_indices.shape[0]):
            cnt = 0
            for j,xj in enumerate(batch_indices[xi,:]):
                xj = xj.item()
                if xi+start_i != xj and distances[xi,j] >= thd and cnt < knn:
                    ei.append(xi+start_i)
                    ej.append(xj)
                    cnt += 1
        del distances,batch_indices
    n = X.shape[0]
    A_knn = sp.csr_matrix((np.ones(len(ei)),(ei,ej)),shape=(n,n))
    A_knn = ((A_knn+A_knn.T)>0).astype(int)
    if device is not None:
        clear_memory(device)
        check_free_memory()
    return A_knn
    
"""
GTDA: our GTDA framework class
nn_model: an instance of NN_model class
labels_to_eval: list of Int, choose which labels to split
smallest_component: Int, the smallest component to stop splitting
overlap: Tuple(Float,Float), overlap ratio, first item represents how much to extend the left side of a bin, second is how much to extend the right side of a bin
extra_lens: Numpy array, any extra lens to use for splitting
"""
def compute_reeb(GTDA,nn_model,labels_to_eval,smallest_component,overlap,extra_lens=None,
    node_size_thd=5,reeb_component_thd=5,test_only=False,alpha=0.5,nsteps=10,nsteps_mixing=10,
    combine_uncertainty=0.1,is_merging=True,split_criteria='diff',split_thd=0,is_normalize=True,
    is_standardize=False,merge_thd=1.0,max_split_iters=50,max_merge_iters=10,nprocs=1,device='cuda',
    degree_normalize_1=1,degree_normalize_2=1):
    t1 = time.time()
    tda = GTDA(nn_model,labels_to_eval)
    print("Preprocess lens")
    M,Ar = tda.build_mixing_matrix(
        alpha=alpha,nsteps=nsteps,selected_nodes=None,extra_lens=extra_lens,
        normalize=is_normalize,standardize=is_standardize,degree_normalize=degree_normalize_1)
    M = torch.tensor(M).to(device)
    A_knn = nn_model.A
    Au = sp.triu(A_knn).tocoo()
    ei,ej = Au.row,Au.col
    e = []
    n = A_knn.shape[0]
    for i in tqdm(range(0,Au.nnz,10000)):
        start_i = i
        end_i = min(start_i+10000,Au.nnz)
        e.append(torch.max(torch.abs((
            M[ei[start_i:end_i],:]-M[ej[start_i:end_i],:])),1)[0].cpu().detach().numpy())
    e = np.concatenate(e)
    selected_edges = np.nonzero(e<merge_thd)[0]
    edges_dists = sp.csr_matrix((e[selected_edges],(ei[selected_edges],ej[selected_edges])),shape=(n,n))
    edges_dists = edges_dists+edges_dists.T
    M = M.cpu().numpy()
    tda.find_reeb_nodes(
        M,Ar,smallest_component=smallest_component,k=1,
        filter_cols=list(range(M.shape[1])),overlap=overlap,component_size_thd=0,
        node_size_thd=node_size_thd,split_criteria=split_criteria,
        split_thd=split_thd,max_iters=max_split_iters)
    if is_merging:
        tda.merge_reeb_nodes(Ar,M,niters=max_merge_iters,node_size_thd=node_size_thd,edges_dists=edges_dists,nprocs=nprocs)
    g_reeb_orig,extra_edges = tda.build_reeb_graph(
        M,Ar,reeb_component_thd=reeb_component_thd,max_iters=max_merge_iters,is_merging=is_merging,edges_dists=edges_dists)
    filtered_nodes = tda.filtered_nodes
    g_reeb = g_reeb_orig[filtered_nodes,:][:,filtered_nodes]
    t2 = time.time()
    time_of_building_reeb_graph = t2-t1
    print(f"Total time for building reeb graph is {time_of_building_reeb_graph} seconds")
    print("Compute mixing rate for each sample")
    tda.generate_node_info(
        nn_model,Ar,g_reeb_orig,extra_edges=extra_edges,class_colors=None,test_only=test_only,combine_uncertainty=combine_uncertainty,
        nsteps=nsteps_mixing,degree_normalize=degree_normalize_2)
    return locals()

def GTDA_analysis(
    GTDA_record,label_to_name=None,is_plotting=True,po1=1.3,po2=1.2,all_class_colors=None,
    return_everything=True,is_computing_coords=True):
    all_figures = []
    tda = GTDA_record['tda']
    g_reeb_orig = GTDA_record['g_reeb_orig']
    Ar = GTDA_record['Ar']
    nn_model = GTDA_record['nn_model']
    labels_to_eval = GTDA_record['labels_to_eval']
    test_only = GTDA_record['test_only']
    pred_labels = np.argmax(nn_model.preds,1)
    test_nodes = np.nonzero(nn_model.test_mask)[0]
    test_accs = np.sum(pred_labels[test_nodes]==nn_model.labels[test_nodes])/len(test_nodes)
    test_accs_corrected = np.sum(tda.corrected_labels[test_nodes]==nn_model.labels[test_nodes])/len(test_nodes)
    test_accs_corrected_simple_mixing = np.sum(tda.corrected_labels_simple_mixing[test_nodes]==nn_model.labels[test_nodes])/len(test_nodes)
    print(test_accs,test_accs_corrected,test_accs_corrected_simple_mixing)
    max_key = np.max(list(tda.final_components_filtered.keys()))
    filtered_nodes = tda.filtered_nodes
    A_sub = g_reeb_orig[filtered_nodes,:][:,filtered_nodes]
    node_sizes = tda.node_sizes[filtered_nodes]
    node_colors_class = tda.node_colors_class[filtered_nodes]
    node_colors_class_truth = tda.node_colors_class_truth[filtered_nodes]
    node_colors_error = tda.node_colors_error[filtered_nodes]
    node_colors_error = (node_colors_error-np.min(node_colors_error))/(np.max(node_colors_error)-np.min(node_colors_error))
    node_colors_error = np.array([to_rgba('red',alpha=i) for i in node_colors_error])
    node_colors_uncertainty = tda.node_colors_uncertainty[filtered_nodes]
    node_colors_uncertainty = (node_colors_uncertainty-np.min(node_colors_uncertainty))/(np.max(node_colors_uncertainty)-np.min(node_colors_uncertainty))
    node_colors_uncertainty = np.array([to_rgba('red',alpha=i) for i in node_colors_uncertainty])
    node_colors_mixing = tda.node_colors_mixing[filtered_nodes]
    node_colors_mixing = (node_colors_mixing-np.min(node_colors_mixing))/(np.max(node_colors_mixing)-np.min(node_colors_mixing))
    node_colors_mixing = np.array([to_rgba('red',alpha=i) for i in node_colors_mixing])
    node_colors_combined = tda.node_colors_combined[filtered_nodes]
    node_colors_combined = (node_colors_combined-np.min(node_colors_combined))/(np.max(node_colors_combined)-np.min(node_colors_combined))
    node_colors_combined = np.array([to_rgba('red',alpha=i) for i in node_colors_combined])
    if all_class_colors is None:
        all_class_colors = sns.color_palette(n_colors=nn_model.preds.shape[1]+2)
    A_sub = A_sub.tocsr()
    pos_all_by_label = {}
    reeb_components = filter_components(A_sub,size_thd=0)[1]
    reeb_components_to_nodes = {}
    for i,reeb_component in enumerate(reeb_components):
        nodes = []
        for reeb_node in reeb_component:
            if filtered_nodes[reeb_node] in tda.final_components_filtered:
                nodes += tda.final_components_filtered[filtered_nodes[reeb_node]]
        if len(nodes) > 0:
            reeb_components_to_nodes[i] = np.unique(nodes)
    reeb_components_by_label = defaultdict(list)
    for key,nodes in reeb_components_to_nodes.items():
        reeb_components_by_label[Counter(pred_labels[nodes]).most_common(1)[0][0]].append(key)
    if is_computing_coords:
        print('compute coordinates for reeb components in each class')
        for selected_label in reeb_components_by_label.keys():
            reeb_components_size = np.array(
                [len(reeb_components[c]) for c in reeb_components_by_label[selected_label]])
            reeb_components_size = reeb_components_size/np.max(reeb_components_size)
            reeb_components_size = reeb_components_size**0.5
            component_pos = []
            items = []
            limits = []
            area = 0
            offset = 0.3
            for i,cid in enumerate(np.argsort(reeb_components_size)[::-1]):
                orig_cid = reeb_components_by_label[selected_label][cid]
                component = reeb_components[orig_cid]
                g_reeb = nx.from_scipy_sparse_matrix(A_sub[component,:][:,component])
                print(len(component))
                pos = nx.kamada_kawai_layout(g_reeb,scale=reeb_components_size[cid])
                ymin = np.min([p[1] for p in pos.values()])
                ymax = np.max([p[1] for p in pos.values()])
                xmin = np.min([p[0] for p in pos.values()])
                xmax = np.max([p[0] for p in pos.values()])
                limits.append((xmin,ymin))
                area += (xmax-xmin+offset)*(ymax-ymin+offset)
                items.append(greedypacker.Item((xmax-xmin+offset), (ymax-ymin+offset)))
                component_pos.append(pos)
            packer = greedypacker.BinManager(
                np.sqrt(area)*po1, np.sqrt(area)*po1, pack_algo='maximal_rectangle', 
                heuristic='bottom_left', rotation=True, sorting_heuristic=False)
            for item in items:
                packer.add_items(item)
            packer.execute()
            pos_all = {}
            for i,cid in enumerate(np.argsort(reeb_components_size)[::-1]):
                orig_cid = reeb_components_by_label[selected_label][cid]
                component = reeb_components[orig_cid]
                pos = component_pos[i]
                item = packer.bins[0].items[i]
                x_offset = (item.x-limits[i][0]+offset/2)
                y_offset = (item.y-limits[i][1]+offset/2)
                for j,nid in enumerate(component):
                    pos_all[nid] = np.array([pos[j][0]+x_offset,pos[j][1]+y_offset])
            pos_all_by_label[selected_label] = pos_all
        items = []
        limits = []
        area = 0
        offset = 0.8
        all_keys = [i[1] for i in sorted([(-1*len(pos_all_by_label[key]),key) for key in pos_all_by_label.keys()])]
        for key in all_keys:
            pos = pos_all_by_label[key]
            ymin = np.min([p[1] for p in pos.values()])
            ymax = np.max([p[1] for p in pos.values()])
            xmin = np.min([p[0] for p in pos.values()])
            xmax = np.max([p[0] for p in pos.values()])
            limits.append((xmin,ymin))
            area += (xmax-xmin+offset)*(ymax-ymin+offset)
            items.append(greedypacker.Item((xmax-xmin+offset), (ymax-ymin+offset)))
        packer = greedypacker.BinManager(
            np.sqrt(area)*po2, np.sqrt(area)*po2, pack_algo='maximal_rectangle', 
            heuristic='bottom_left', rotation=True, sorting_heuristic=True)
        for item in items:
            packer.add_items(item)
        packer.execute()
        for i,key in enumerate(all_keys):
            item = packer.bins[0].items[i]
            x_offset = (item.x-limits[i][0]+offset/2)
            y_offset = (item.y-limits[i][1]+offset/2)
            for j in pos_all_by_label[key].keys():
                pos_all_by_label[key][j][0] += x_offset
                pos_all_by_label[key][j][1] += y_offset
        pos = {}
        for curr in pos_all_by_label.values():
            pos.update(curr)
        print('combine all nodes coordinates')
        df = pd.DataFrame(
            data=[(
                filtered_nodes[key],
                pos[key][0],
                pos[key][1],
                25*node_sizes[key]**0.5,
                tuple(node_colors_class[key,:])) for key in range(len(pos.keys()))],
            columns=['id', 'x', 'y', 'sizes', 'colors'])
    if is_plotting:
        g_reeb = nx.from_scipy_sparse_matrix(A_sub)
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        for node_id in tqdm(range(df.shape[0])):
            if np.max(node_colors_class[node_id,:]) == np.sum(node_colors_class[node_id,:]):
                ax.scatter(
                    [df.x.values[node_id]],
                    [df.y.values[node_id]],
                    edgecolors=[0,0,0,0.2],
                    linewidths=0.5,
                    color=all_class_colors[np.argmax(node_colors_class[node_id,:])],
                    s=df.sizes.values[node_id]
                )
            else:
                draw_pie(node_colors_class[node_id,:], 
                    df.x.values[node_id], 
                    df.y.values[node_id],
                    df.sizes.values[node_id], 
                    [0,0,0,0.2],
                    0.5,
                    ax=ax,
                    class_colors=all_class_colors)
        A_sub = A_sub.tocoo()
        for ei,ej in zip(A_sub.row,A_sub.col):
            ax.plot(
                [pos[ei][0],pos[ej][0]],
                [pos[ei][1],pos[ej][1]],
                c='black',zorder=-10,linewidth=0.2,alpha=0.5)
        ax.axis('off')
        all_figures.append((fig,ax))
        plt.show()
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        for node_id in tqdm(range(df.shape[0])):
            if np.max(node_colors_class_truth[node_id,:]) == np.sum(node_colors_class_truth[node_id,:]):
                ax.scatter(
                    [df.x.values[node_id]],
                    [df.y.values[node_id]],
                    edgecolors=[0,0,0,0.2],
                    linewidths=0.5,
                    color=all_class_colors[np.argmax(node_colors_class_truth[node_id,:])],
                    s=df.sizes.values[node_id]
                )
            else:
                draw_pie(node_colors_class_truth[node_id,:], 
                    df.x.values[node_id], 
                    df.y.values[node_id],
                    df.sizes.values[node_id], 
                    [0,0,0,0.2],
                    0.5,
                    ax=ax,
                    class_colors=all_class_colors)
        A_sub = A_sub.tocoo()
        for ei,ej in zip(A_sub.row,A_sub.col):
            ax.plot(
                [pos[ei][0],pos[ej][0]],
                [pos[ei][1],pos[ej][1]],
                c='black',zorder=-10,linewidth=0.2,alpha=0.5)
        patches = []
        for label_to_eval in labels_to_eval:
            patches.append(
                Line2D(
                    [0],[0],marker='o',markerfacecolor=all_class_colors[label_to_eval],
                    label=f"{label_to_name[label_to_eval]}",color='w',markersize=20))
        ax.legend(
            handles=patches,fontsize=20,loc="upper center",bbox_to_anchor=(0.5, 1.1),ncol=3)
        ax.axis('off')
        all_figures.append((fig,ax))
        plt.show()
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        nx.draw(
            g_reeb,pos,node_size=df.sizes.values,
            node_color=node_colors_error,
            width=0.2,ax=ax,edgecolors=[0,0,0,0.2],linewidths=0.5,
            edge_color=[0,0,0,0.5])
        ax.set_title('colored by error rate')
        all_figures.append((fig,ax))
        plt.show()
        plt.close("all")
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        nx.draw(
            g_reeb,pos,node_size=df.sizes.values,
            node_color=node_colors_uncertainty,
            width=0.2,ax=ax,edgecolors=[0,0,0,0.2],linewidths=0.5,
            edge_color=[0,0,0,0.5])
        ax.set_title('colored by uncertainty rate')
        all_figures.append((fig,ax))
        plt.show()
        plt.close("all")
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        nx.draw(
            g_reeb,pos,node_size=df.sizes.values,
            node_color=node_colors_mixing,
            width=0.2,ax=ax,edgecolors=[0,0,0,0.2],linewidths=0.5,
            edge_color=[0,0,0,0.5])
        ax.set_title('colored by mixing rate')
        all_figures.append((fig,ax))
        plt.show()
        plt.close("all")
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        nx.draw(
            g_reeb,pos,node_size=df.sizes.values,
            node_color=node_colors_combined,
            width=0.2,ax=ax,edgecolors=[0,0,0,0.2],linewidths=0.5,
            edge_color=[0,0,0,0.5])
        ax.set_title('colored by all combined rate')
        all_figures.append((fig,ax))
        plt.show()
        plt.close("all")
        pre_labels = np.argmax(nn_model.preds,1)
        labels = nn_model.labels
        recalls_uncertainty = compute_recall(
            labels,pre_labels,tda.sample_colors_uncertainty,test_only=test_only,test_nodes=test_nodes)
        recalls_mixing = compute_recall(
            labels,pre_labels,tda.sample_colors_mixing,test_only=test_only,test_nodes=test_nodes)
        recalls_mixing_combined = compute_recall(
            labels,pre_labels,tda.sample_colors_combined,test_only=test_only,test_nodes=test_nodes)
        recalls_random = compute_recall(
            labels,pre_labels,np.random.rand(len(labels)),test_only=test_only,test_nodes=test_nodes)
        recalls_truth = compute_recall(
            labels,pre_labels,labels!=pre_labels,test_only=test_only,test_nodes=test_nodes)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        tmp = len(recalls_mixing)
        ax.plot(range(tmp),recalls_mixing[0:tmp])
        tmp = len(recalls_mixing)
        ax.plot(range(tmp),recalls_mixing_combined[0:tmp])
        tmp = len(recalls_mixing)
        ax.plot(range(tmp),recalls_uncertainty[0:tmp])
        tmp = len(recalls_mixing)
        ax.plot(range(tmp),recalls_random[0:tmp])
        tmp = len(recalls_mixing)
        ax.plot(range(tmp),recalls_truth[0:tmp])
        ax.legend(['GTDA','GTDA combined','uncertainty','random','ground truth'],fontsize=15)
        ax.set_ylabel('Recall',fontsize=15)
        ax.set_xlabel('# samples checked',fontsize=15)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
        plt.show()
        all_figures.append((fig,ax))
        plt.show()
        test_nodes = np.nonzero(nn_model.test_mask)[0]
        sample_colors_error = np.copy(tda.sample_colors_error)
        sample_colors_error[test_nodes] = 0
        node_colors_error = np.zeros(max_key+1)
        for key in tda.final_components_filtered.keys():
            component = np.array(tda.final_components_filtered[key])
            if len(component) > 0:
                node_colors_error[key] = np.mean(sample_colors_error[component])
        node_colors_error = node_colors_error[filtered_nodes]
        node_colors_error = np.array([to_rgba('red',alpha=i) for i in node_colors_error])
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        nx.draw(
            g_reeb,pos,node_size=df.sizes.values,
            node_color=node_colors_error,
            width=0.2,ax=ax,edgecolors=[0,0,0,0.2],linewidths=0.5,
            edge_color=[0,0,0,0.5])
        ax.set_title('colored by train error rate')
        all_figures.append((fig,ax))
        plt.show()
        plt.close("all")
        max_key = np.max(list(tda.final_components_filtered.keys()))
    print('compute mistakes')
    mistakes = (pred_labels != nn_model.labels)
    reeb_mistakes = {}
    reeb_mixing = {}
    for i,reeb_component in tqdm(enumerate(reeb_components)):
        if i in reeb_components_to_nodes:
            reeb_mistakes[i] = (
                np.sum(mistakes[reeb_components_to_nodes[i]]),np.sum(mistakes[reeb_components_to_nodes[i]])/len(reeb_components_to_nodes[i]))
    reeb_node_mistakes = {}
    for reeb_node in tda.final_components_filtered.keys():
        nodes = tda.final_components_filtered[reeb_node]
        reeb_node_mistakes[reeb_node] = (
                np.sum(mistakes[nodes]),np.sum(mistakes[nodes])/len(nodes))
    print('prepare to return')
    if return_everything:
        return locals()
    else:
        return {
            'tda': tda,
            'reeb_mistakes': reeb_mistakes,
            'reeb_mixing': reeb_mixing,
            'pred_labels': pred_labels,
            'labels': labels,
            'reeb_components_to_nodes': reeb_components_to_nodes,
            'df': df,
            'node_colors_class': node_colors_class,
            'node_colors_class_truth': node_colors_class_truth,
            'reeb_node_mistakes': reeb_node_mistakes,
        }


def draw_pie(dist, 
             xpos, 
             ypos, 
             size, 
             edgecolors,
             linewidths,
             ax=None,
             class_colors=None,alpha=1,zorder=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    if class_colors is None:
        class_colors = sns.color_palette(n_colors=len(dist)+2)

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()

    for i, (r1, r2) in enumerate(zip(pie[:-1], pie[1:])):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])
        if alpha == 1:
            ax.scatter(
                [xpos], [ypos], marker=xy, s=size, color=class_colors[i],
                edgecolors=edgecolors, linewidths=linewidths, zorder=zorder)
        else:
            ax.scatter(
                [xpos], [ypos], marker=xy, s=size, color=class_colors[i],
                edgecolors=edgecolors, linewidths=linewidths, alpha=alpha, zorder=zorder)

    return ax

def plot_image_grid(img_list,text="",filename=""):
    scale = 0.1
    nrows = int(np.sqrt(len(img_list)))+1
    ncols = nrows
    fig,ax = plt.subplots(figsize=(20,20),dpi=72)
    row = 0
    col = ncols-1
    for i,img in enumerate(img_list):
        if row == nrows:
            row = 0
            col -= 1
        offset = (row*scale,row*scale+scale,col*scale,col*scale+scale)
        ax.imshow(img,extent=offset)
        row += 1
    ax.set_xlim((0,scale*nrows))
    ax.set_ylim((0,scale*nrows))
    ax.set_title(text)
    plt.close(fig)
    fig.savefig(filename)

def generate_mixing_matrix(A_knn,shape,init_y,alpha,nsteps,nodes):
    init_mixing = np.zeros(shape)
    for node in nodes:
        init_mixing[node,init_y[node]] = 1
    degs = np.sum(A_knn,0)
    dinv = 1/degs
    dinv[dinv==np.inf] = 0
    Dinv = sp.spdiags(dinv,0,A_knn.shape[0],A_knn.shape[0])
    An = Dinv@A_knn
    total_mixing_all = np.copy(init_mixing)
    for i in tqdm(range(nsteps)):
        total_mixing_all = (1-alpha)*init_mixing + alpha*An@total_mixing_all
    return total_mixing_all


def plot_reeb_component(
    nodes,G,df_all,node_colors,pos_all,is_plotting_error=False,is_plotting_legend=True,
    labels_to_eval=None,all_class_colors=None,label_to_name=None,train_nodes=[],alphas=None,
    ax=None,fig=None,linealpha=0.5,edge_zorder=1,node_zorder=2,nodelinewidths=0.5,bbox_to_anchor=(0.5,1.1),
    fontsize=20,linewidth=0.2,nodeedgecolors=0.2):
    df = df_all.iloc[nodes]
    G = G.tocsr()
    G = G[nodes,:][:,nodes]
    pos = {i:pos_all[nodes[i]] for i in range(len(nodes))}
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
    if is_plotting_error:
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        nx.draw(
            nx.from_scipy_sparse_matrix(G),pos,node_size=df.sizes.values,
            node_color=node_colors[nodes,:],
            width=linewidth,ax=ax,edgecolors=[0,0,0,nodeedgecolors],linewidths=nodelinewidths,
            edge_color=[0,0,0,linealpha])
    else:
        xmin,xmax = df.x.values.min()-0.1,df.x.values.max()+0.1
        ymin,ymax = df.y.values.min()-0.1,df.y.values.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        for i in tqdm(range(df.shape[0])):
            node_id = nodes[i]
            if np.max(node_colors[node_id,:]) == np.sum(node_colors[node_id,:]):
                ax.scatter(
                    [df.x.values[i]],
                    [df.y.values[i]],
                    edgecolors=[0,0,0,0.2],
                    linewidths=nodelinewidths,
                    color=to_rgba(all_class_colors[np.argmax(node_colors[node_id,:])],alphas[i] if alphas else 1),
                    s=df.sizes.values[i],zorder=node_zorder
                )
            else:
                draw_pie(node_colors[node_id,:], 
                    df.x.values[i], 
                    df.y.values[i],
                    df.sizes.values[i], 
                    [0,0,0,0.2],
                    nodelinewidths,
                    ax=ax,
                    class_colors=all_class_colors, 
                    alpha= alphas[i] if alphas else 1,
                    zorder=node_zorder)
        G = G.tocoo()
        for ei,ej in zip(G.row,G.col):
            ax.plot(
                [pos[ei][0],pos[ej][0]],
                [pos[ei][1],pos[ej][1]],
                c='black',linewidth=linewidth,
                alpha=linealpha,zorder=edge_zorder)
        if is_plotting_legend:
            patches = []
            for label_to_eval in labels_to_eval:
                patches.append(
                    Line2D(
                        [0],[0],marker='o',markerfacecolor=all_class_colors[label_to_eval],
                        label=f"{label_to_name[label_to_eval]}",color='w',markersize=20))
            ax.legend(
                handles=patches,fontsize=fontsize,loc="upper center",bbox_to_anchor=bbox_to_anchor,ncol=5)
        ax.axis('off')
    return fig,ax

def clear_memory(device):
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def check_free_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(t-r-a,(t-r-a)/t)  # free inside reserved

def rotate_coords(origin, point, angle):
    angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy])

def rotate_component_coords(xcoords,ycoords,angle):
    xcenter = np.mean(xcoords)
    ycenter = np.mean(ycoords)
    for key in range(len(xcoords)):
        new_coords = rotate_coords(
            [xcenter,ycenter], [xcoords[key],ycoords[key]], angle)
        xcoords[key] = new_coords[0]
        ycoords[key] = new_coords[1]
    return xcoords,ycoords

def mirror_component_coords(xcoords,ycoords,axis):
    xcenter = np.mean(xcoords)
    ycenter = np.mean(ycoords)
    for key in range(len(xcoords)):
        if axis == 0:
            xcoords[key] = 2*ycenter - xcoords[key]
        else:
            ycoords[key] = 2*xcenter - ycoords[key]
    return xcoords,ycoords

def extend_coords(origin, point, scale):
    ox, oy = origin
    px, py = point
    qx = scale*(px-ox)+ox
    qy = scale*(py-oy)+oy
    return np.array([qx, qy])

def plot_subgraph(node_colors,pos,subg,
    all_class_colors=None,label_to_name=None,labels_to_eval=None,plot_legend=True,nodesize=40,
    linealpha=0.5,ax=None,fig=None,edgealpha=0.5,fontsize=20,ncol=5,loc='upper center'):
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
    for i in tqdm(pos.keys()):
        node_id = i
        ax.scatter(
            pos[node_id][0],
            pos[node_id][1],
            linewidths=0,
            c=node_colors[node_id:(node_id+1)],
            s=nodesize
        )
    G = subg.tocoo()
    for ei,ej in zip(G.row,G.col):
        ax.plot(
            [pos[ei][0],pos[ej][0]],
            [pos[ei][1],pos[ej][1]],
            c='black',zorder=-10,linewidth=0.2,alpha=edgealpha)
    if plot_legend and all_class_colors is not None and label_to_name is not None and labels_to_eval is not None:
        patches = []
        for label_to_eval in labels_to_eval:
            patches.append(
                Line2D(
                    [0],[0],marker='o',markerfacecolor=all_class_colors[label_to_eval],
                    label=f"{label_to_name[label_to_eval]}",color='w',markersize=20))
        ax.legend(
            handles=patches,fontsize=fontsize,loc=loc,ncol=ncol)
    ax.axis('off')
    return fig,ax