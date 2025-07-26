from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag



def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio, f1 = [], [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))
        f1.append(F1(precision[0], recall[0]))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc,
            'f1': np.array(f1)}


def evaluate_one_user(x, train_user_set, test_user_set, train_user_set_neg):
    res = []
    for item in x:
        rating = item[0]
        u = item[1]

        try:
            training_items = train_user_set[u]
            training_items_neg = train_user_set_neg[u]
        except Exception:
            training_items = []
            training_items_neg = []

        user_pos_test = test_user_set[u]

        all_items = set(range(0, n_items))
        test_items = list(all_items - set(training_items)-set(training_items_neg))

        if args.test_flag == 'part':
            r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        else:
            r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

        res.append(get_performance(user_pos_test, r, auc, Ks))

    return res



def evaluate(model, user_dict,user_dict_neg, n_params, is_train=False):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': np.zeros(len(Ks)),
              'f1':np.zeros(len(Ks))}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    if is_train:
        user_set = user_dict['train_user_set']
    else:
        user_set = user_dict['test_user_set']
    user_set_train = user_dict['train_user_set']


    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    train_user_set_neg = user_dict_neg['train_user_set_neg']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    users = list(user_set.keys())
    n_users = len(users)
    n_user_batchs = n_users // u_batch_size + 1

    count = 0

    entity_gcn_emb, user_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]



        if batch_test_flag:#"use gpu or not"
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)


                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)

                i_g_embeddings = entity_gcn_emb[item_batch]


                i_rate_batch = model.rating(u_g_embeddings, i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:

            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = evaluate_one_user(user_batch_rating_uid, train_user_set, test_user_set, train_user_set_neg)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_users
            result['recall'] += re['recall']/n_users
            result['ndcg'] += re['ndcg']/n_users
            result['hit_ratio'] += re['hit_ratio']/n_users
            result['auc'] += re['auc']/n_users
            result['f1'] += re['f1'] / n_users

    assert count == n_users
    pool.close()
    return result


def test(model, user_dict,user_dict_neg, n_params):

    test_result = evaluate(model, user_dict,user_dict_neg, n_params, is_train=False)

    train_result = []

    return {'test': test_result, 'train': train_result}



