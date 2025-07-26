import numpy as np
from sklearn.metrics import roc_auc_score
import math
from torch.autograd import no_grad
import torch

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def rank(num_user, user_item_inter, mask_items, result, u_result, hat_u_result, i_result, hat_i_result, is_training,
         step, topk, model_name):
    user_tensor = result[:num_user]
    print(len(user_tensor))
    print(u_result)
    print(hat_u_result)
    item_tensor = result[num_user:]
    user_rep = u_result
    hat_user_rep = hat_u_result  # F.normalize(hat_u_result)
    item_rep = i_result
    hat_item_rep = hat_i_result  # F.normalize(hat_i_result)
    print(model_name)

    start_index = 0
    end_index = num_user if step == None else step
    all_index_of_rank_list = torch.LongTensor([])

    while end_index <= num_user and start_index < end_index:
        temp_user_tensor = user_tensor[start_index:end_index]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

        temp_user_rep = user_rep[start_index:end_index]  # len*dim
        u_i_e_score = torch.matmul(temp_user_rep, item_rep.t())  # len*num_item
        ex_uie_score = torch.mean(u_i_e_score, dim=0, keepdim=True)  # 1*num_item

        temp_hat_user_rep = hat_user_rep[start_index:end_index]  # len*dim
        temp_score = torch.matmul(temp_hat_user_rep, hat_item_rep.t())  # len*num_item
        temp_score = (u_i_e_score - ex_uie_score) * torch.sigmoid(temp_score)  # len*num_item * (len*num_item)

        score_matrix += temp_score

        if is_training is False:

            for row, col in user_item_inter.items():

                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - num_user
                    score_matrix[row][col] = 1e-15

        print('score',score_matrix)
        _, index_of_rank_list = torch.topk(score_matrix, topk)
        print(index_of_rank_list)
        all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu() + num_user), dim=0)

        start_index = end_index
        if end_index + step < num_user:
            end_index += step
        else:
            end_index = num_user
    print("all_index_of_rank_list:")
    print(all_index_of_rank_list)
    return all_index_of_rank_list


def full_accuracy(val_data, all_index_of_rank_list, user_item_inter, is_training, topk):
    length = 0
    precision = recall = ndcg = 0.0
    sum_num_hit = 0
    for data in val_data:
        user = data[0]
        pos_items = set(data[1:])
        num_pos = len(pos_items)
        if num_pos == 0:
            continue
        length += 1
        items_list = all_index_of_rank_list[user].tolist()
        items = set(items_list)

        num_hit = len(pos_items.intersection(items))
        sum_num_hit += num_hit
        precision += float(num_hit / topk)
        recall += float(num_hit / num_pos)
        ndcg_score = 0.0
        max_ndcg_score = 0.0
        for i in range(min(num_pos, topk)):
            max_ndcg_score += 1 / math.log2(i + 2)

        if max_ndcg_score == 0:
            continue
        for i, temp_item in enumerate(items_list):
            if temp_item in pos_items:
                ndcg_score += 1 / math.log2(i + 2)
        ndcg += ndcg_score / max_ndcg_score

    return precision / length, recall / length, ndcg / length


def full_ranking(epoch, model, data, user_item_inter, mask_items, is_training, step, topk, model_name, prefix,
                 writer=None):
    print(prefix + ' start...')
    model.eval()
    with no_grad():

        all_index_of_rank_list = rank(model.num_u, user_item_inter, mask_items, model.result, model.u_result,
                                      model.hat_u_result, model.i_result, model.hat_i_result, is_training, step, topk,
                                      model_name)
        precision, recall, ndcg_score = full_accuracy(data, all_index_of_rank_list, user_item_inter, is_training, topk)

        print(
            '---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
                epoch, precision, recall, ndcg_score))

        return [precision, recall, ndcg_score]


