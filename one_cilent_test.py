import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from utils import args_parser
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from model import Net
from Update import LocalUpdate
from Fed import FedAvg
from predict import test
import random
import gc
import pandas as pd


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    with open('dataset/num.json', encoding='utf8') as i_f:
        sta = json.load(i_f)
    exer_n = sta["problem_n"]
    knowledge_n = sta["skill_n"]
    student_n = sta["user_n"]
    school_n = sta["school_n"]
    log_n = sta["all_n"]

    # load data
    data_file = 'dataset/train_set.json'
    with open(data_file, encoding='utf8') as i_f:
        dataset_train = json.load(i_f)

    # student_n = max(stu['user_id'] for stu in dataset_train)
    # knowledge_n = max(stu['knowledge_code'] for stu in dataset_train)
    # exer_n = max(stu['exer_id'] for stu in dataset_train)

    data_file = 'dataset/test_set.json'
    with open(data_file, encoding='utf8') as i_f:
        dataset_test = json.load(i_f)


    args.local_ep = 1
    args.model = "NCD"
    args.lr = 0.02
    args.epochs = 20

    # each cilent
    for school in range(school_n):

        # data
        dataset_train_sch = [stu for stu in dataset_train if stu['school_id'] == school]
        # random.shuffle(dataset_train)
        dict_users = range(len(dataset_train_sch))
        dataset_test_sch = [stu for stu in dataset_test if stu['school_id'] == school]

        # build model
        net_glob = Net(student_n, exer_n, knowledge_n)
        net_glob = net_glob.to(args.device)
        # print(net_glob)
        net_glob.train()

        # copy weights
        w_glob = net_glob.state_dict()

        # training
        loss_train = []
        # best loss
        net_best = None
        epoch_best = None
        loss_best = None
        acc_best = 0
        rmse_best = None
        auc_best = None

        print("School {:3d} Training ".format(school))

        for iter in range(args.epochs):

            local = LocalUpdate(args=args, lr=args.lr, dataset=dataset_train, idxs=dict_users)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            net_glob.load_state_dict(w)

            print('Round {:3d}, Average loss {:.4f}'.format(iter + 1, loss))
            loss_train.append(loss)

            # validation
            acc_test, rmse_test, auc_test = test(net_glob, dataset_test, args)
            print("Testing accuracy: {:.4f}, Testing rmse: {:.4f}, Testing auc: {:.4f}".format(acc_test, rmse_test, auc_test))

            if acc_test > acc_best or (acc_test == acc_best and auc_test > auc_best):
                net_best = net_glob.state_dict()
                epoch_best = iter + 1
                loss_best = loss
                acc_best = acc_test
                rmse_best = rmse_test
                auc_best = auc_test

        net_glob.eval()
        net_glob.load_state_dict(net_best)
        acc_train, rmse_train, auc_train = test(net_glob, dataset_train, args)
        acc_test, rmse_test, auc_test = test(net_glob, dataset_test, args)
        print("School {:3d} : Best result at epoch {:3d}".format(school, epoch_best))
        print("Training accuracy: {:.4f}, Training rmse: {:.4f}, Training auc: {:.4f}".format(acc_train, rmse_train,
                                                                                              auc_train))
        print("Testing accuracy: {:.4f}, Testing rmse: {:.4f}, Testing auc: {:.4f}".format(acc_test, rmse_test,
                                                                                           auc_test))