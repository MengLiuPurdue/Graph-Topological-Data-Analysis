from .GTDA_utils import find_components
from bisect import bisect_right
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, Counter
import itertools
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import copy
import time
import faiss

def is_overlap(x,y):
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if (xi[0] >= yi[0] and xi[0] <= yi[1]) or (yi[0] >= xi[0] and yi[0] <= xi[1]):
            continue
        else:
            return False
    return True

class GTDA(object):
    def __init__(self,nn_model,labels_to_eval):
        self.A = nn_model.A
        self.preds = np.copy(nn_model.preds)
        self.labels_to_eval = copy.copy(labels_to_eval)
    
    def _compute_bin_lbs(self,inner_id,overlap,col_id,nbins):
        if inner_id >= nbins or inner_id < 0:
            return float("inf")
        new_inner_id = inner_id
        curr_lb = self.pre_lbs[col_id]+self.bin_sizes[col_id]*inner_id
        flag = False
        for _ in range(int(np.ceil(overlap))):
            if new_inner_id == 0:
                flag = False
                break
            new_inner_id = new_inner_id-1
            curr_lb -= self.bin_sizes[col_id]
            flag = True
        if flag:
            curr_lb += (np.ceil(overlap)-overlap)*self.bin_sizes[col_id]
        return curr_lb
    
    def _compute_bin_ubs(self,inner_id,overlap,col_id,nbins):
        if inner_id >= nbins or inner_id < 0:
            return -1*float("inf")
        new_inner_id = inner_id
        curr_ub = self.pre_lbs[col_id]+self.bin_sizes[col_id]*(1+inner_id)
        flag = False
        for _ in range(int(np.ceil(overlap))):
            if new_inner_id == nbins-1:
                flag = False
                break
            new_inner_id = new_inner_id+1
            curr_ub += self.bin_sizes[col_id]
            flag = True
        if flag:
            curr_ub -= (np.ceil(overlap)-overlap)*self.bin_sizes[col_id]
        return curr_ub

    def build_mixing_matrix(
        self,alpha=0.5,nsteps=3,normalize=True,extra_lens=None,standardize=False,degree_normalize=1):
        Ar = (self.A>0).astype(np.float64)
        degs = np.sum(Ar,0)
        dinv = 1/degs
        dinv[dinv==np.inf] = 0
        Dinv = sp.spdiags(dinv,0,Ar.shape[0],Ar.shape[0])
        if degree_normalize == 1:
            An = Dinv@Ar
        elif degree_normalize == 2:
            An = Ar@Dinv
        elif degree_normalize == 3:
            An = np.sqrt(Dinv)@Ar@np.sqrt(Dinv)
        else:
            An = Ar
        total_mixing_all = np.copy(self.preds)
        init_mixing = np.copy(self.preds)
        if extra_lens is not None:
            total_mixing_all = np.hstack([total_mixing_all,extra_lens])
            init_mixing = np.hstack([init_mixing,extra_lens])
        for i in tqdm(range(nsteps)):
            total_mixing_all = (1-alpha)*init_mixing + alpha*An@total_mixing_all
        selected_col = self.labels_to_eval
        if extra_lens is not None:
            selected_col += list(range(self.preds.shape[1],total_mixing_all.shape[1]))
        M = total_mixing_all[:,selected_col].copy()
        if standardize:
            for i in range(M.shape[1]):
                M[:,i] = (M[:,i]-np.mean(M[:,i]))/np.std(M[:,i])
        if normalize:
            for i in range(M.shape[1]):
                if np.max(M[:,i]) != np.min(M[:,i]):
                    M[:,i] = (M[:,i]-np.min(M[:,i]))/(np.max(M[:,i])-np.min(M[:,i]))
        return M,Ar
    
    def _clustering_single_col_pyramid(self,
        M,filter_cols,nbins,lbs=None,ubs=None):
        self.pre_lbs = np.zeros(len(filter_cols))
        self.pre_ubs = np.zeros(len(filter_cols))
        self.bin_sizes = np.zeros(len(filter_cols))
        self.lbs = np.min(M,0) if lbs is None else lbs
        self.ubs = np.max(M,0) if ubs is None else ubs
        for i,col in enumerate(filter_cols):
            self.pre_lbs[i] = self.lbs[col]
            self.pre_ubs[i] = self.ubs[col]
            self.bin_sizes[i] = (self.ubs[col]-self.lbs[col])/nbins
    
    def _find_bins_pyramid(self,M,filter_cols,overlap,nbins):
        bin_nums = 0
        bin_key_map = {}
        bin_map = {}
        bins = defaultdict(list)
        all_assignments = [[[] for _ in range(len(filter_cols))] for _ in range(M.shape[0])]
        for j,col in enumerate(filter_cols):
            bin_size = self.bin_sizes[j]
            inner_id = np.floor((M[:,col]-self.pre_lbs[j])/bin_size).astype(np.int64)
            boundary = np.nonzero(M[:,col] == self.pre_ubs[j])[0]
            inner_id[boundary] = nbins-1
            bin_ids = inner_id
            for i,bin_id in enumerate(bin_ids):
                all_assignments[i][j].append(bin_id)
            bin_lbs = []
            bin_ubs = []
            for t in range(nbins):
                bin_lbs.append(self._compute_bin_lbs(t,overlap[0],j,nbins))
                bin_ubs.append(self._compute_bin_ubs(t,overlap[1],j,nbins))
            bin_lbs = np.array(bin_lbs)
            bin_ubs = np.array(bin_ubs)
            for offset in range(1,1+int(np.ceil(max(overlap)))):
                new_inner_id = inner_id+offset
                valid_ids = np.nonzero((new_inner_id>=0)*(new_inner_id<nbins))[0]
                valid_bin_ids = new_inner_id[valid_ids]
                filtered_ids = np.nonzero(M[valid_ids,col] >= bin_lbs[valid_bin_ids])[0]
                valid_ids = valid_ids[filtered_ids]
                valid_bin_ids = valid_bin_ids[filtered_ids]
                for i,valid_id in enumerate(valid_ids):
                    all_assignments[valid_id][j].append(valid_bin_ids[i])
                new_inner_id = inner_id-offset
                valid_ids = np.nonzero((new_inner_id>=0)*(new_inner_id<nbins))[0]
                valid_bin_ids = new_inner_id[valid_ids]
                filtered_ids = np.nonzero(M[valid_ids,col] <= bin_ubs[valid_bin_ids])[0]
                valid_ids = valid_ids[filtered_ids]
                valid_bin_ids = valid_bin_ids[filtered_ids]
                for i,valid_id in enumerate(valid_ids):
                    all_assignments[valid_id][j].append(valid_bin_ids[i])
        for i in range(M.shape[0]):
            for bin_key in itertools.product(*all_assignments[i]):
                if bin_key not in bin_key_map:
                    bin_key_map[bin_key] = bin_nums
                    bin_map[bin_nums] = bin_key
                    bins[bin_nums].append(i)
                    bin_nums += 1
                else:
                    assigned_id = bin_key_map[bin_key]
                    bins[assigned_id].append(i)
        return bins,bin_map

    def filtering(
        self,M,filter_cols,nbins=2,overlap=(0.05,0.05),**kwargs):
        self._clustering_single_col_pyramid(M,filter_cols,nbins,**kwargs)
        return self._find_bins_pyramid(M,filter_cols,overlap,nbins)

    def graph_clustering(self,G,bins,component_size_thd=10):
        graph_clusters = defaultdict(list)
        for key in bins.keys():
            points = bins[key]
            if len(points) < component_size_thd:
                continue
            Gr = G[points,:][:,points].copy()
            _,components = find_components(Gr,component_size_thd)
            for component in components:
                graph_clusters[key].append([points[node] for node in component])
        return graph_clusters
    
    def check_bin(self,curr_bin,bin_diam_thd,num_check,check_k_nodes,M,filter_cols,curr_indices):
        if len(curr_bin) < check_k_nodes:
            return False
        d = len(filter_cols)
        index = faiss.IndexFlatL2(d)
        index.add(M[curr_bin,:][:,filter_cols].copy().astype(np.float32))
        xqs = [np.random.choice([0,1],size=len(filter_cols)) for _ in range(num_check)]
        xqs_inv = [1-xq for xq in xqs]
        xqs = np.array([[curr_indices[i,c] for i,c in enumerate(xq)] for xq in xqs],dtype=np.float32)
        xqs_inv = np.array([[curr_indices[i,c] for i,c in enumerate(xq)] for xq in xqs_inv],dtype=np.float32)
        D, _ = index.search(xqs, check_k_nodes)
        Dinv, _ = index.search(xqs_inv, check_k_nodes)
        diam = np.sqrt(np.sum((curr_indices[:,1] - curr_indices[:,0])**2))
        Dxqs = np.sqrt(np.mean(D,1))
        Dxqs_inv = np.sqrt(np.mean(Dinv,1))
        return np.any(diam>bin_diam_thd*(Dxqs+Dxqs_inv))

    def find_reeb_nodes(self,M,Ar,
        filter_cols=None,nbins_pyramid=2,overlap=(0.5,0.5),node_size_thd=10,
        smallest_component=50,component_size_thd=0,split_criteria="diff",split_thd=0.01,max_iters=50):
        self.component_records_all = {}
        self.final_components_all = {}
        self.component_records = {}
        self.final_components = {}
        self.component_counts = [0]
        self.split_lens = {}
        self.component_id_map = defaultdict(list)
        _,components = find_components(Ar,size_thd=0)
        curr_level = []
        num_final_components = 0
        num_total_components = 0
        for component in components:
            self.component_records[num_total_components] = component
            if len(component) > smallest_component:
                curr_level.append(num_total_components)
            else:
                self.final_components[num_final_components] = component
                num_final_components += 1
            num_total_components += 1
        self.component_counts.append(len(self.component_records))
        iters = 0
        def worker(M_sub,G_sub,component_id,nbins_pyramid,overlap,component_size_thd,
            split_thd):
            if split_criteria == 'std':
                diffs = np.std(M_sub,0)
            else:
                diffs = np.max(M_sub,0) - np.min(M_sub,0)
            col_to_filter = np.argmax(diffs)
            largest_diff = diffs[col_to_filter]
            if largest_diff < split_thd:
                return [list(range(M_sub.shape[0]))],component_id,largest_diff,col_to_filter
            else:
                bins,_ = self.filtering(
                    M_sub,[col_to_filter],nbins=nbins_pyramid,overlap=overlap)
                graph_clusters = self.graph_clustering(G_sub,bins,
                        component_size_thd=component_size_thd)
                return graph_clusters,component_id,largest_diff,col_to_filter
        slice_columns = True
        if len(np.setdiff1d(range(M.shape[1]),filter_cols)) == 0:
            slice_columns = False
        while len(curr_level) > 0 and iters < max_iters:
            iters += 1
            print(f"Iteration {iters}")
            print(f"{len(curr_level)} components to split")
            sizes = []
            new_level = []
            all_G_sub = []
            all_M_sub = []
            t1 = time.time()
            for component_id in curr_level:
                component = np.array(self.component_records[component_id])
                G_sub = Ar[component,:][:,component].copy()
                if slice_columns:
                    M_sub = M[component,:][:,filter_cols].copy()
                else:
                    M_sub = M[component,:].copy()
                all_G_sub.append(G_sub)
                all_M_sub.append(M_sub)
            t2 = time.time()
            print(f"Grouping took {t2-t1} seconds")
            process_order = sorted([(-1*all_M_sub[i].shape[0],i) for i in range(len(all_M_sub))])
            t1 = time.time()
            min_largest_diff = float("inf")
            max_largest_diff = -1*float("inf")
            for _,i in tqdm(process_order):
                ret = worker(
                    all_M_sub[i],all_G_sub[i],curr_level[i],
                    nbins_pyramid,overlap,component_size_thd,split_thd)
                graph_clusters,component_id,largest_diff,col_to_filter = ret
                min_largest_diff = min(min_largest_diff,largest_diff)
                max_largest_diff = max(max_largest_diff,largest_diff)
                self.split_lens[component_id] = col_to_filter
                component = np.array(self.component_records[component_id])
                if largest_diff < split_thd:
                    self.final_components[num_final_components] = component.tolist()
                    num_final_components += 1
                    num_total_components += 1
                    sizes.append(len(component))
                else:
                    for components in graph_clusters.values():
                        for new_component in components:
                            sizes.append(len(new_component))
                            self.component_records[num_total_components] = component[new_component].tolist()
                            self.component_id_map[component_id].append(num_total_components)
                            if (len(new_component) > smallest_component):
                                new_level.append(num_total_components)
                            else:
                                self.final_components[num_final_components] = component[new_component].tolist()
                                num_final_components += 1
                            num_total_components += 1
            print(f"Min/max largest difference: {min_largest_diff}, {max_largest_diff}")
            print("New components sizes:")
            print(Counter(sizes))
            curr_level = new_level
            t2 = time.time()
            print(f"Splitting took {t2-t1} seconds")
            self.component_counts.append(len(self.component_records))
        if len(curr_level) > 0:
            for i in curr_level:
                self.final_components[num_final_components] = self.component_records[i]
                num_final_components += 1
        self._remove_duplicate_components()
        self._filter_tiny_components(Ar,node_size_thd)
    
    def _remove_duplicate_components(self):
        all_c = sorted([
            sorted(self.final_components[key]) for key in self.final_components.keys()])
        filtered_c = list(k for k,_ in itertools.groupby(all_c))
        self.final_components_unique = {i:c for i,c in enumerate(filtered_c)}


    def _filter_tiny_components(self,Ar,node_size_thd):
        nodes = []
        for val in self.final_components_unique.values():
            nodes += val
        print("Number of samples included before filtering:", len(set(nodes)))
        all_keys = self.final_components_unique.keys()
        filtered_keys = []
        removed_keys = []
        self.node_assignments = [set() for i in range(Ar.shape[0])]
        self.node_assignments_tiny_components = [set() for i in range(Ar.shape[0])]
        for key in all_keys:
            if len(self.final_components_unique[key]) > node_size_thd:
                for node in self.final_components_unique[key]:
                    self.node_assignments[node].add(key)
                filtered_keys.append(key)
            else:
                for node in self.final_components_unique[key]:
                    self.node_assignments_tiny_components[node].add(key)
                removed_keys.append(key)
        self.final_components_removed = {
            key:self.final_components_unique[key] for key in removed_keys}
        self.final_components_filtered = {
            key:self.final_components_unique[key] for key in filtered_keys}
        nodes = []
        for val in self.final_components_filtered.values():
            nodes += val
        nodes = list(set(nodes))
        print("Number of samples included after filtering:", len(nodes))

    def merge_reeb_nodes(self,Ar,M,niters=1,node_size_thd=10,edges_dists=None,nprocs=10):
        num_components = len(self.final_components_filtered)+len(self.final_components_removed)
        def worker(tmp_edges_dists,nodes,k1):
            closest_neigh = -1
            neighs = tmp_edges_dists.indices
            valid_neighs = np.setdiff1d(neighs,nodes)
            tmp_edges_dists = tmp_edges_dists[:,valid_neighs]
            if tmp_edges_dists.data.shape[0] > 0:
                closest_neigh_id = np.argmin(tmp_edges_dists.data)
                closest_neigh = tmp_edges_dists.indices[closest_neigh_id]
                closest_neigh = valid_neighs[closest_neigh]
            return closest_neigh,k1
        modified = True
        self.edges_to_merge = []
        for _ in range(niters):
            if modified:
                modified = False
            else:
                break
            merging_ei = []
            merging_ej = []
            keys_to_check = self.final_components_removed.keys()
            print("Merge reeb nodes...")
            processed_list = Parallel(n_jobs=nprocs)(
                delayed(worker)(
                    edges_dists[self.final_components_removed[k1],:],
                    self.final_components_removed[k1],k1) for k1 in tqdm(keys_to_check))
            for closest_neigh,k1 in processed_list:
                if closest_neigh != -1:
                    components_to_connect = list(self.node_assignments[closest_neigh])+list(self.node_assignments_tiny_components[closest_neigh])
                    if len(components_to_connect) > 0:
                        sizes = []
                        for c in components_to_connect:
                            if c in self.final_components_filtered:
                                sizes.append(len(self.final_components_filtered[c]))
                            else:
                                sizes.append(len(self.final_components_removed[c]))
                        component_to_connect = components_to_connect[np.argmin(sizes)]
                        merging_ei.append(component_to_connect)
                        merging_ej.append(k1)
                        self.edges_to_merge.append((component_to_connect,k1))
                        modified = True
            merging_map = sp.csr_matrix((np.ones(len(merging_ei)),(merging_ei,merging_ej)),shape=(num_components,num_components))
            merging_map = (merging_map+merging_map.T)>0
            self._merging_tiny_nodes(merging_map,node_size_thd)
    
    def _merging_tiny_nodes(self,merging_map,node_size_thd):
        keys_to_remove = set()
        components_to_merge = find_components(merging_map,size_thd=1)[1]
        for component_to_merge in components_to_merge:
            component_to_connect = component_to_merge[0]
            for k in component_to_merge:
                if k in self.final_components_filtered:
                    component_to_connect = k
            new_component = []
            for k in component_to_merge:
                if k == component_to_connect:
                    continue
                nodes = self.final_components_removed[k]
                if component_to_connect in self.final_components_filtered:
                    self.final_components_filtered[component_to_connect] += nodes
                    self.final_components_filtered[component_to_connect] = list(set(self.final_components_filtered[component_to_connect]))
                    keys_to_remove.add(k)
                    for node in nodes:
                        self.node_assignments[node].add(component_to_connect)
                        self.node_assignments_tiny_components[node].remove(k)
                else:
                    new_component += nodes
            if component_to_connect not in self.final_components_filtered:
                new_component += self.final_components_removed[component_to_connect]
                new_component = list(set(new_component))
                if len(new_component) > node_size_thd:
                    for k in component_to_merge:
                        nodes = self.final_components_removed[k]
                        keys_to_remove.add(k)
                        for node in nodes:
                            self.node_assignments[node].add(component_to_connect)
                            self.node_assignments_tiny_components[node].remove(k)
                    self.final_components_filtered[component_to_connect] = new_component
                else:
                    for k in component_to_merge:
                        nodes = self.final_components_removed[k]
                        if k != component_to_connect:
                            keys_to_remove.add(k)
                        for node in nodes:
                            self.node_assignments_tiny_components[node].remove(k)
                            self.node_assignments_tiny_components[node].add(component_to_connect)
                    self.final_components_removed[component_to_connect] = new_component
        for k in keys_to_remove:
            del self.final_components_removed[k]
        nodes = []
        for val in self.final_components_filtered.values():
            nodes += val
        nodes = list(set(nodes))
        print("Number of samples included after merging:", len(nodes))
    
    def generate_node_info(
        self,nn_model,Ar,g_reeb,extra_edges=None,class_colors=None,alpha=0.5,nsteps=10,
        pre_labels=None,known_nodes=None,degree_normalize=1):
        if known_nodes is None:
            known_mask_np = nn_model.train_mask+nn_model.val_mask
            known_nodes = np.nonzero(known_mask_np)[0]
        else:
            known_mask_np = np.zeros(Ar.shape[0],dtype=bool)
            known_mask_np[known_nodes] = True
        labels = nn_model.labels
        if pre_labels is None:
            pre_labels = np.argmax(nn_model.preds,1)
        max_key = np.max(list(self.final_components_filtered.keys()))
        self.node_sizes = np.zeros(max_key+1)
        self.node_colors_class = np.zeros((max_key+1,nn_model.preds.shape[1]))
        self.node_colors_class_truth = np.zeros((max_key+1,nn_model.preds.shape[1]))
        self.node_colors_error = np.zeros(max_key+1)
        self.node_colors_uncertainty = np.zeros(max_key+1)
        self.node_colors_mixing = np.zeros(max_key+1)
        self.sample_colors_mixing = np.zeros(nn_model.preds.shape[0])
        uncertainty = 1-np.max(nn_model.preds,1)
        self.sample_colors_uncertainty = uncertainty
        self.sample_colors_error = np.zeros(nn_model.preds.shape[0])
        if class_colors is None:
            class_colors = sns.color_palette(n_colors=nn_model.preds.shape[1])
        reeb_components = find_components(g_reeb,size_thd=0)[1]
        ei,ej = [],[]
        for reeb_component in reeb_components:
            for reeb_node in reeb_component:
                if reeb_node in self.final_components_filtered:
                    nodes = self.final_components_filtered[reeb_node]
                    nodes = list(set(nodes))
                    mapping = {i:k for i,k in enumerate(nodes)}
                    sub_A = Ar[nodes,:][:,nodes].tocoo()
                    for i,j in zip(sub_A.row,sub_A.col):
                        ei.append(mapping[i])
                        ej.append(mapping[j])
        if extra_edges is not None:
            ei += extra_edges[0]
            ej += extra_edges[1]
        self.A_reeb = sp.csr_matrix((np.ones(len(ei)),(ei,ej)),shape=Ar.shape)
        self.A_reeb = self.A_reeb+self.A_reeb.T
        self.A_reeb = (self.A_reeb>0).astype(int)
        training_node_labels = np.zeros((Ar.shape[0],nn_model.preds.shape[1]))
        for node in known_nodes:
            training_node_labels[node,labels[node]] = 1
        degs = np.sum(Ar,0)
        dinv = 1/degs
        dinv[dinv==np.inf] = 0
        Dinv = sp.spdiags(dinv,0,Ar.shape[0],Ar.shape[0])
        degs = np.sum(self.A_reeb,0)
        dinv = 1/degs
        dinv[dinv==np.inf] = 0
        Dinv = sp.spdiags(dinv,0,self.A_reeb.shape[0],self.A_reeb.shape[0])
        if degree_normalize == 1:
            self.An = Dinv@self.A_reeb
        elif degree_normalize == 2:
            self.An = self.A_reeb@Dinv
        elif degree_normalize == 3:
            self.An = np.sqrt(Dinv)@self.A_reeb@np.sqrt(Dinv)
        else:
            self.An = self.A_reeb
        self.total_mixing_all = np.copy(training_node_labels)
        for i in range(nsteps):
            self.total_mixing_all = (1-alpha)*training_node_labels + alpha*self.An@self.total_mixing_all
        for i in range(self.total_mixing_all.shape[0]):
            if np.sum(self.total_mixing_all[i,:]) > 0:
                d = self.total_mixing_all[i,pre_labels[i]]/np.sum(self.total_mixing_all[i,:])
                self.sample_colors_mixing[i] = 1-d
            else:
                self.sample_colors_mixing[i] = uncertainty[i]
            self.sample_colors_error[i] = 1-(pre_labels[i]==labels[i])
            self.sample_colors_uncertainty[i] = uncertainty[i]
        for key in self.final_components_filtered.keys():
            component = np.array(self.final_components_filtered[key])
            self.node_sizes[key] = len(component)
            component_label_cnt = Counter(labels[component])
            for l,lc in component_label_cnt.items():
                self.node_colors_class_truth[key,l] = lc
            component_label_cnt = Counter(pre_labels[component])
            for l,lc in component_label_cnt.items():
                self.node_colors_class[key,l] = lc
            if len(component) > 0:
                self.node_colors_error[key] = np.mean(self.sample_colors_error[component])
                self.node_colors_uncertainty[key] = np.mean(self.sample_colors_uncertainty[component])
                self.node_colors_mixing[key] = np.mean(self.sample_colors_mixing[component])
    
               
    def build_reeb_graph(self,M,Ar,reeb_component_thd=10,max_iters=10,is_merging=True,edges_dists=None):
        all_edge_index = [[], []]
        extra_edges = [[],[]]
        print("Build reeb graph...")
        reeb_dim = np.max(list(self.final_components_filtered.keys()))+1
        ei,ej = [],[]
        for key,c in self.final_components_filtered.items():
            ei += [key]*len(c)
            ej += c
        bipartite_g = sp.csr_matrix((np.ones(len(ei)),(ei,ej)),shape=(reeb_dim,M.shape[0]))
        bipartite_g_t = bipartite_g.T.tocsr()
        ei,ej = [],[]
        for i in tqdm(self.final_components_filtered.keys()):
            neighs = set(bipartite_g_t[bipartite_g[i,:].indices].indices)
            neighs.remove(i)
            neighs = list(neighs)
            all_edge_index[0] += [i]*len(neighs)
            all_edge_index[1] += neighs
        A_tmp = sp.csr_matrix(
            (np.ones(len(all_edge_index[1])),(all_edge_index[0],all_edge_index[1])),shape=(
                reeb_dim,reeb_dim))
        A_tmp = ((A_tmp+A_tmp.T)>0).astype(np.float64)
        _,components_left1 = find_components(A_tmp,size_thd=0)
        components_removed = []
        self.filtered_nodes = []
        for c in components_left1:
            if len(c) > reeb_component_thd:
                self.filtered_nodes += c
            else:
                for node in c:
                    if node in self.final_components_filtered:
                        components_removed.append(c)
                        break
        curr_iter = 0
        modified = True
        while modified and is_merging and len(components_removed) > 0 and curr_iter < max_iters:
            modified = False
            curr_iter += 1
            for component_removed in tqdm(components_removed):
                nodes_removed = []
                for key in component_removed:
                    nodes_removed += self.final_components_filtered[key]
                tmp_edges_dists = edges_dists[nodes_removed]
                neighs = tmp_edges_dists.indices
                valid_neighs = np.setdiff1d(neighs,nodes_removed)
                tmp_edges_dists = tmp_edges_dists[:,valid_neighs]
                key_to_connect = -1
                closest_neigh = -1
                if tmp_edges_dists.data.shape[0] > 0:
                    closest_neigh_id = np.argmin(tmp_edges_dists.data)
                    closest_neigh = tmp_edges_dists.indices[closest_neigh_id]
                    closest_neigh = valid_neighs[closest_neigh]
                    node_to_connect = nodes_removed[bisect_right(tmp_edges_dists.indptr,closest_neigh_id)-1]
                    key_to_connect = np.min(list(self.node_assignments[node_to_connect].intersection(component_removed)))
                if closest_neigh != -1:
                    components_to_connect = list(self.node_assignments[closest_neigh])
                    if len(components_to_connect) > 0:
                        sizes = []
                        for c in components_to_connect:
                            if c in self.final_components_filtered:
                                sizes.append(len(self.final_components_filtered[c]))
                            else:
                                sizes.append(len(self.final_components_removed[c]))
                        component_to_connect = components_to_connect[np.argmin(sizes)]
                        all_edge_index[0].append(key_to_connect)
                        all_edge_index[1].append(component_to_connect)
                        extra_edges[0].append(node_to_connect)
                        extra_edges[1].append(closest_neigh)
                        modified = True
            A_tmp = sp.csr_matrix(
                (np.ones(len(all_edge_index[1])),(all_edge_index[0],all_edge_index[1])),shape=(
                    reeb_dim,reeb_dim))
            A_tmp = ((A_tmp+A_tmp.T)>0).astype(np.float64)
            _,components_left1 = find_components(A_tmp,size_thd=0)
            components_removed = []
            self.filtered_nodes = []
            for c in components_left1:
                if len(c) > reeb_component_thd:
                    self.filtered_nodes += c
                else:
                    for node in c:
                        if node in self.final_components_filtered:
                            components_removed.append(c)
                            break
        nodes = []
        self.filtered_nodes = np.intersect1d(self.filtered_nodes,list(self.final_components_filtered.keys()))
        for i in self.filtered_nodes:
            component = np.array(self.final_components_filtered[i])
            nodes += component.tolist()
        nodes = list(set(nodes))
        print("Number of samples included after merging reeb components:", len(set(nodes)))
        return A_tmp,extra_edges