'''
This is the PyTorch Implementation for model VRKG4Rec (WSDM'23)

Contact me via email (lulingyun@hust.edu.cn), if you have any questions.
'''
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random

import torch
import numpy as np
from time import time

import torchvision
from prettytable import PrettyTable
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test

from utils.helper import early_stopping
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.model import SRVSKG




n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
# 保存每个 epoch 的结果
recall_values = []
epochs = []
ndcg_values = []
precision_values = []


def get_feed_dict(train_entity_pairs, start, end, train_user_set,train_user_set_net):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, pop_item in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]

                # if neg_item not in train_user_set[user] and neg_items not in train_user_set_net[user]:
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)

        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set,train_user_set_net)).to(device)
    return feed_dict




if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""


    train_cf, test_cf, user_dict, n_params, graph, triplets, relation_dict, mat_list, kg_data,  L_eigs, indices, user_dict_neg,train_cf_neg,test_cf_neg = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list, ua_adj_mean_mat = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']



    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    if args.model == 'my_model_sign':
        train_cf_pairs_neg = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf_neg], np.int32))
        test_cf_pairs_neg = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf_neg], np.int32))

    """kg data"""
    train_kg_pairs = torch.LongTensor(np.array([[kg[0], kg[1], kg[2]] for kg in triplets], np.int32))

    print("define model ...")
    """define model"""
    model = SRVSKG(n_params, args, indices, L_eigs, graph, mean_mat_list, ua_adj_mean_mat)




    print("define optimizer ...")
    """define optimizer"""

    cur_best_pre_0 = 0
    stopping_step = 0
    best_epoch = 0
    should_stop = False



    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]



        """ training """

        if args.model == 'my_model_sign':
            train_sign_s = time()
            mf_loss_total_sign = model.train_sign(train_cf, train_cf_neg, args)
            train_sign_e = time()
            optimizer = torch.optim.Adam(model.my_parameters_my, lr=args.lr)
            train_cf_s = time()
            mf_loss_total, s = 0, 0
            while s + args.batch_size <= len(train_cf):
                cf_batch = get_feed_dict(train_cf_pairs,
                                                  s, s + args.batch_size,
                                                  user_dict['train_user_set'],
                                                  user_dict_neg['train_user_set_neg'])
                # batch_loss, mf_loss, _ = model(cf_batch)
                mf_loss = model.computer_my_model(cf_batch)
                optimizer.zero_grad()
                mf_loss.backward()
                optimizer.step()
                mf_loss_total += mf_loss.item()
                s += args.batch_size
            train_cf_e = time()




        if epoch % 10 == 9 or epoch == 1:#每10个epoch测试一次
            # print("start testing ...")
            """testing"""
            test_s_t = time()
            model.eval()

            result = test(model, user_dict,user_dict_neg, n_params)

            ret = result['test']

            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "testing time", "recall", "ndcg", "precision",
                                     "hit_ratio", "auc", "f1"]
            train_res.add_row(
                [epoch, train_cf_e - train_cf_s, test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'],
                 ret['hit_ratio'], ret['auc'], ret['f1']]
            )
            print(train_res)

            cur_best_pre_0, stopping_step, should_stop, best_epoch = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                    stopping_step, best_epoch, epoch,
                                                                                    expected_order='acc',
                                                                                    flag_step=10)
            if args.save:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, args.out_dir + 'model_' + args.dataset + "_" +str(epoch) + '.ckpt')
            if should_stop:
                break

            """save weight"""
            # if ret['recall'][0] == cur_best_pre_0 and args.save:
                # torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            print('my_model using time %.4f, training loss at epoch %d: %.4f' % (train_cf_e - train_cf_s, epoch, mf_loss_total))
            print('sigfomer using time %.4f, training loss at epoch %d: %.4f' % (train_sign_e - train_sign_s, epoch, mf_loss_total_sign))



