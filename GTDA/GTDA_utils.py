import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import numpy as np
from collections import Counter
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from datetime import datetime
import pickle
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from matplotlib.lines import Line2D
import gc
import os
import os.path as osp
import time
from rembg.detect import ort_session
from rembg import remove

def extend_coords(origin, point, scale):
    ox, oy = origin
    px, py = point
    qx = scale*(px-ox)+ox
    qy = scale*(py-oy)+oy
    return np.array([qx, qy])

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


def find_components(A,size_thd=100,verbose=False):
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
    components = find_components(A,size_thd=0)[1]
    largest_component_id = np.argmax([len(c) for c in components])
    return components[largest_component_id]

class NN_model(object):
    def __init__(self):
        self.preds = None
        self.A = None
        self.labels = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

# convert any graph with feature matrix to InMemoryDataset
class data_generator(InMemoryDataset):
    def __init__(self,G,X,labels,name,root_path="dataset/",
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

def SPoC(model,dataloaders,key_map=None,is_normalize=False,indices_to_ignore=None,pooling="max",device='cuda'):
    model = model.to(device)
    X = None
    y = []
    preds = []
    for dataloader in dataloaders:
        cnt = len(dataloader)
        for inputs,labels in tqdm(dataloader,total=cnt):
            with torch.set_grad_enabled(False):
                tmp = model.bn1(model.conv1(inputs.to(device)))
                tmp = model.relu(tmp)
                tmp = model.maxpool(tmp)
                tmp = model.layer1(tmp)
                tmp = model.layer2(tmp)
                tmp = model.layer3(tmp)
                tmp = model.layer4(tmp)
                if pooling == "max":
                    pooling_layer = torch.nn.AdaptiveMaxPool2d(output_size=(1,1)).cuda()
                    tmp = pooling_layer(tmp)
                else:
                    tmp = model.avgpool(tmp)
                X.append(tmp.flatten(start_dim=2).sum(dim=2).cpu().detach().numpy())
                y += labels.numpy().tolist()
                tmp = torch.flatten(tmp, 1)
                tmp = model.fc(tmp)
                if indices_to_ignore is not None:
                    tmp[:,indices_to_ignore] = -1*float('inf')
                preds.append(F.softmax(tmp,dim=1).cpu().detach().numpy())
    if is_normalize:
        X = normalize(np.concatenate(X))
    else:
        X = np.concatenate(X)
    preds = np.concatenate(preds)
    if key_map is not None:
        preds = preds[:,key_map]
    if device != 'cpu':
        inputs.cpu()
        torch.cuda.empty_cache()
    return X,y,preds

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

"""
This function builds a KNN graph with 'X', which is num_samples-by-num_embedding_dim, using GPU. 
It only supports cosine similarity. Consider to reduce batch_size or batch_size_training if GPU memory is not big enough.
"""
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
    node_size_thd=5,reeb_component_thd=5,alpha=0.5,nsteps_preprocess=5,nsteps_mixing=10,is_merging=True,
    split_criteria='diff',split_thd=0,is_normalize=True,is_standardize=False,merge_thd=1.0,max_split_iters=200,
    max_merge_iters=10,nprocs=1,device='cuda',degree_normalize_preprocess=1,degree_normalize_mixing=1):
    t1 = time.time()
    gtda = GTDA(nn_model,labels_to_eval)
    print("Preprocess lens")
    M,Ar = gtda.build_mixing_matrix(
        alpha=alpha,nsteps=nsteps_preprocess,extra_lens=extra_lens,normalize=is_normalize,
        standardize=is_standardize,degree_normalize=degree_normalize_preprocess)
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
    gtda.find_reeb_nodes(
        M,Ar,smallest_component=smallest_component,
        filter_cols=list(range(M.shape[1])),overlap=overlap,component_size_thd=0,
        node_size_thd=node_size_thd,split_criteria=split_criteria,
        split_thd=split_thd,max_iters=max_split_iters)
    if is_merging:
        gtda.merge_reeb_nodes(Ar,M,niters=max_merge_iters,node_size_thd=node_size_thd,edges_dists=edges_dists,nprocs=nprocs)
    g_reeb_orig,extra_edges = gtda.build_reeb_graph(
        M,Ar,reeb_component_thd=reeb_component_thd,max_iters=max_merge_iters,is_merging=is_merging,edges_dists=edges_dists)
    filtered_nodes = gtda.filtered_nodes
    g_reeb = g_reeb_orig[filtered_nodes,:][:,filtered_nodes]
    t2 = time.time()
    time_of_building_reeb_graph = t2-t1
    print(f"Total time for building reeb graph is {time_of_building_reeb_graph} seconds")
    print("Compute mixing rate for each sample")
    gtda.generate_node_info(
        nn_model,Ar,g_reeb_orig,extra_edges=extra_edges,class_colors=None,
        nsteps=nsteps_mixing,degree_normalize=degree_normalize_mixing)
    GTDA_record = {
        "g_reeb": g_reeb,
        "gtda": gtda,
        "extra_edges": extra_edges,
        "time_of_building_reeb_graph": time_of_building_reeb_graph,
        "M": M,
    }
    return GTDA_record

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

def plot_reeb_component(
    G,df,node_colors,is_plotting_error=False,is_plotting_legend=False,
    labels_to_eval=None,all_class_colors=None,label_to_name=None,train_nodes=[],alphas=None,
    ax=None,fig=None,linealpha=0.5,edge_zorder=1,node_zorder=2,nodelinewidths=0.5,bbox_to_anchor=(0.5,1.1),
    fontsize=20,linewidth=0.2,nodeedgecolors=0.2):
    G = G.tocsr()
    pos = {}
    xcoords = df.x.values
    ycoords = df.y.values
    for i in range(len(ycoords)):
        pos[i] = np.array([xcoords[i],ycoords[i]])
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,12),dpi=72)
    if is_plotting_error:
        xmin,xmax = xcoords.min()-0.1,xcoords.max()+0.1
        ymin,ymax = ycoords.min()-0.1,ycoords.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        nx.draw(
            nx.from_scipy_sparse_matrix(G),pos,node_size=df.sizes.values,
            node_color=node_colors,
            width=linewidth,ax=ax,edgecolors=[0,0,0,nodeedgecolors],linewidths=nodelinewidths,
            edge_color=[0,0,0,linealpha])
    else:
        xmin,xmax = xcoords.min()-0.1,xcoords.max()+0.1
        ymin,ymax = ycoords.min()-0.1,ycoords.max()+0.1
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        for i in tqdm(range(df.shape[0])):
            if np.max(node_colors[i,:]) == np.sum(node_colors[i,:]):
                ax.scatter(
                    [xcoords[i]],
                    [ycoords[i]],
                    edgecolors=[0,0,0,0.2],
                    linewidths=nodelinewidths,
                    color=to_rgba(all_class_colors[np.argmax(node_colors[i,:])],alphas[i] if alphas else 1),
                    s=df.sizes.values[i],zorder=node_zorder
                )
            else:
                draw_pie(node_colors[i,:], 
                    xcoords[i], 
                    ycoords[i],
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

def align_images(start_pos,end_pos,scale,images,ax,nrows,is_padding,flipped=False):
    num_images_embeded = 0
    angle = np.arctan2(end_pos[1]-start_pos[1],end_pos[0]-start_pos[0])
    if (angle >= np.pi/4 and angle <= 3*np.pi/4) or (angle >= -3*np.pi/4 and angle <= -np.pi/4):
        num_images = int(np.ceil(abs((end_pos[1]-start_pos[1])/scale)))
        scale = abs(end_pos[1]-start_pos[1])/num_images
        offset = (scale,0)
    else:
        num_images = int(np.ceil(abs((end_pos[0]-start_pos[0])/scale)))
        scale = abs(end_pos[0]-start_pos[0])/num_images
        offset = (0,scale)
    curr_i = 0
    if is_padding:
        start_i = -1
    else:
        start_i = 0
    all_img_align_pos = []
    all_align_imgs = []
    for i in range(start_i,num_images):
        if curr_i >= len(images):
                break
        for row in range(nrows):
            if curr_i >= len(images):
                break
            x_center = start_pos[0]+(i+1)*(end_pos[0]-start_pos[0])/num_images
            y_center = start_pos[1]+(i+1)*(end_pos[1]-start_pos[1])/num_images
            if row % 2 == 0:
                x_center -= (row//2)*offset[0]
                y_center -= (row//2)*offset[1]
            else:
                x_center += (row//2+1)*offset[0]
                y_center += (row//2+1)*offset[1]
            if flipped:
                curr_pos = [
                    x_center+scale/2,x_center-scale/2,
                    y_center-scale/2,y_center+scale/2]
            else:
                curr_pos = [
                    x_center-scale/2,x_center+scale/2,
                    y_center-scale/2,y_center+scale/2]
            num_images_embeded += 1
            all_img_align_pos.append(curr_pos)
            all_align_imgs.append(curr_i)
            curr_i += 1
    for i,curr_i in enumerate(all_align_imgs[::-1]):
        ax.imshow(images[curr_i],extent=all_img_align_pos[i])
    return num_images_embeded


def remove_img_bg(img):
    session = ort_session("u2net")
    new_img = remove(img,session=session)
    return new_img