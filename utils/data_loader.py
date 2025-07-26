import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import pickle
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')
import pandas as pd
from utils.parser import parse_args
import csv

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
train_item_set = defaultdict(list)
train_user_set_neg = defaultdict(list)
test_user_set_neg = defaultdict(list)
train_item_set_neg = defaultdict(list)


def read_cf(data_name):
    inter_mat = data_name[:, :2]

    return np.array(inter_mat)





def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_entities = int(n_entities)
    n_nodes = n_entities + n_users
    n_nodes = int(n_nodes)
    n_relations = max(triplets[:, 1]) + 1

    return triplets



def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict, ua_triplets):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)


    ua_mat = np.array(ua_triplets[:, [0, 2]])

    vals = [1.] * len(ua_mat)
    ua_adj_mat = sp.coo_matrix((vals, (ua_mat[:, 0], ua_mat[:, 1])), shape=(n_nodes, n_nodes))
    ua_adj_mean_mat = _si_norm_lap(ua_adj_mat)



    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]

    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users,
                       n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list, ua_adj_mean_mat


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'
    print('reading train and test user-item set ...')

    n_user, n_item, train_data, train_data_neg, test_data, test_data_neg = load_rating(args)

    global n_users, n_items
    n_users = n_user
    n_items = n_item
    n_users = int(n_users)
    n_items = int(n_items)


    train_cf = read_cf(train_data)
    test_cf = read_cf(test_data)
    train_cf_neg = read_cf(train_data_neg)
    test_cf_neg = read_cf(test_data_neg)


    for u_id, i_id in train_cf:
        train_user_set[int(u_id)].append(int(i_id))
        train_item_set[int(i_id)].append(int(u_id))
    for u_id, i_id in test_cf:
        test_user_set[int(u_id)].append(int(i_id))

    for u_id, i_id in train_cf_neg:
        train_user_set_neg[int(u_id)].append(int(i_id))
        train_item_set_neg[int(i_id)].append(int(u_id))
    for u_id, i_id in test_cf_neg:
        test_user_set_neg[int(u_id)].append(int(i_id))

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)
    ua_triplets = np.load(directory + 'ua_triplets.npy')
    kg_list = np.column_stack((triplets[:, 0], triplets[:, 2]))

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list, ua_adj_mean_mat = build_sparse_relational_graph(relation_dict,
                                                                                                ua_triplets)

    L_eigs, indices = cul_L_eigs(train_data, train_data_neg, args)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
        'train_item_set': train_item_set
    }


    user_dict_neg = {
        'train_user_set_neg': train_user_set_neg,
        'test_user_set_neg': test_user_set_neg,
        'train_item_set_neg': train_item_set_neg

        }
    print('n_users:', n_users)
    print('n_items', n_items)
    return train_cf, test_cf, user_dict, n_params, graph, triplets, relation_dict, \
            [adj_mat_list, norm_mat_list, mean_mat_list, ua_adj_mean_mat], kg_list,  L_eigs, indices,user_dict_neg,train_cf_neg,test_cf_neg






def load_rating(args):
    print('reading training file and testing file ...')
    directory = 'data/' + args.dataset
    train_data_new = np.loadtxt(directory + '/train.txt', dtype=float)
    test_data_new = np.loadtxt(directory + '/test.txt', dtype=float)
    train_data_new = train_data_new.astype(int)
    test_data_new = test_data_new.astype(int)
    rating_np = np.concatenate((train_data_new, test_data_new), axis=0)
    n_user = max(set(rating_np[:, 0])) + 1
    n_item = max(set(rating_np[:, 1])) + 1
    print('n_user:',n_user)
    print('n_item',n_item)
    train_pos_data = train_data_new[train_data_new[:, 2] >= args.offset]
    train_neg_data = train_data_new[train_data_new[:, 2] < args.offset]
    test_pos_data = test_data_new[test_data_new[:, 2] >= args.offset]
    test_neg_data = test_data_new[test_data_new[:, 2] < args.offset]

    return n_user, n_item, train_pos_data, train_neg_data, test_pos_data, test_neg_data


def cul_L(train_user, train_item, args):
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    A = torch.sparse_coo_tensor(
        torch.cat([
            torch.stack([train_user, train_item + n_users]),
            torch.stack([train_item + n_users, train_user])], dim=1),
        torch.ones(train_user.shape[0] * 2).to(device),
        torch.Size([n_users+n_items, n_users+n_items]))

    degree = A.sum(dim=1).to_dense()

    D = degree.float()
    D[D == 0.] = 1.
    D1 = torch.sparse_coo_tensor(
        torch.arange(n_users+n_items, device=device).unsqueeze(0).repeat(2, 1),
        D ** (-1 / 2),
        torch.Size([n_users+n_items, n_users+n_items]))
    D2 = torch.sparse_coo_tensor(
        torch.arange(n_users+n_items, device=device).unsqueeze(0).repeat(2, 1),
        D ** (-1 / 2),
        torch.Size([n_users+n_items, n_users+n_items]))
    tildeA = torch.sparse.mm(torch.sparse.mm(D1, A), D2)

    D3 = torch.sparse_coo_tensor(
        torch.arange(n_users+n_items, device=device).unsqueeze(0).repeat(2, 1),
        torch.ones(n_users+n_items, device=device),
        torch.Size([n_users+n_items, n_users+n_items]))
    L = D3 - tildeA
    return L

def cul_L_eigs(train_pos_data, train_neg_data, args):
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    train_pos_user = torch.from_numpy(train_pos_data[:, 0]).to(device)
    train_pos_item = torch.from_numpy(train_pos_data[:, 1]).to(device)
    train_neg_user = torch.from_numpy(train_neg_data[:, 0]).to(device)
    train_neg_item = torch.from_numpy(train_neg_data[:, 1]).to(device)
    L_pos = cul_L(train_pos_user, train_pos_item, args)
    L_neg = cul_L(train_neg_user, train_neg_item, args)
    L = (L_pos + args.alpha_sign * L_neg) / (1 + args.alpha_sign)

    _, L_eigs = sp.linalg.eigs(
        sp.csr_matrix(
            (L._values().cpu(), L._indices().cpu()),
            (n_users+n_items, n_users+n_items)),
        k=args.eigs_dim,
        which='SR')
    L_eigs = torch.tensor(L_eigs.real).to(device)
    L_eigs = F.layer_norm(L_eigs, normalized_shape=(args.eigs_dim,))

    indices = torch.cat([
        torch.stack([train_pos_user, train_pos_item + n_users]),
        torch.stack([train_pos_item + n_users, train_pos_user]),
        torch.stack([train_neg_user, train_neg_item + n_users]),
        torch.stack([train_neg_item + n_users, train_neg_user])], dim=1)
    sorted_indices = torch.argsort(indices[0, :])
    indices = indices[:, sorted_indices]


    return L_eigs, indices




