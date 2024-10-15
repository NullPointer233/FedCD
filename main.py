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
from predict import test,client_test
import random
import gc
import pandas as pd


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    with open('assist12_73/num.json', encoding='utf8') as i_f:
        sta = json.load(i_f)
    exer_n = sta["problem_n"]
    knowledge_n = sta["skill_n"]
    student_n = sta["user_n"]
    school_n = sta["school_n"]
    log_n = sta["all_n"]

    args.num_users = student_n
    args.num_items = exer_n
    args.num_clients = school_n


    # load data
    data_file = 'assist12_73/train_set.json'
    with open(data_file, encoding='utf8') as i_f:
        dataset_train = json.load(i_f)
    # a=1
    # random.shuffle(dataset_train)
    # a=1
    data_file = 'assist12_73/test_set.json'
    with open(data_file, encoding='utf8') as i_f:
        dataset_test = json.load(i_f)
    # allcate users to schools
    dict_users_train, dict_users_test = {}, {}
    for sch in range(school_n):
        dict_users_train[sch] = set()
        dict_users_test[sch] = set()
    train_idx = 0
    for stu in dataset_train:
        dict_users_train[stu['school_id']].add(train_idx)
        train_idx += 1
    test_idx = 0
    for stu in dataset_test:
        dict_users_test[stu['school_id']].add(test_idx)
        test_idx += 1

    # test if one student in several school in trainset
    school_students = {}
    for stu in dataset_train:
        school_id = stu['school_id']
        student_id = stu['user_id']
        if school_id not in school_students:
            school_students[school_id] = set()
        school_students[school_id].add(student_id)
    has_overlap = False
    school_ids = list(school_students.keys())
    for i in range(len(school_ids)):
        for j in range(i + 1, len(school_ids)):
            school_i = school_ids[i]
            school_j = school_ids[j]
            overlap = school_students[school_i].intersection(school_students[school_j])
            if overlap:
                has_overlap = True
                print(f"School {school_i} and School {school_j} have overlapping students: {overlap}")
    if not has_overlap:
        print("No overlapping students found among schools.")


    # build model
    net_glob = Net(student_n, exer_n, knowledge_n)
    net_glob = net_glob.to(args.device)
    print(net_glob)
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
    rmse_best = 1
    auc_best = 0

    args.all_clients = True

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_clients)]


    for iter in range(args.epochs):

        loss_locals = []
        if not args.all_clients:
            w_locals = []

        m = max(int(args.frac * school_n), 1)
        # idxs_users = np.random.choice(range(stu_i), m, replace=False)
        idxs_users = np.array(range(school_n)) #
        #idxs_users = np.array(range(2))

        a =1

        for idx in idxs_users:

            net_glob.load_state_dict(w_locals[idx])
            local = LocalUpdate(args=args, lr=args.lr, dataset=dataset_train, idxs=dict_users_train[idx])
            w, loss = local.train(net=net_glob.to(args.device))


            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_locals = FedAvg(w_locals, dict_users_train, school_students)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.4f}'.format(iter+1, loss_avg))
        loss_train.append(loss_avg)

        # validation
        acc_test, rmse_test, auc_test = test(net_glob, w_locals, dataset_test, dict_users_test, args)
        print("Testing accuracy: {:.4f}, Testing rmse: {:.4f}, Testing auc: {:.4f}".format(acc_test, rmse_test, auc_test))


        if acc_test > acc_best or (acc_test == acc_best and auc_test > auc_best) or (acc_test == acc_best and auc_test == auc_best and rmse_test < rmse_best):
            net_best = copy.deepcopy(w_locals)
            epoch_best = iter + 1
            loss_best = loss_avg
            acc_best = acc_test
            rmse_best = rmse_test
            auc_best = auc_test

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    # testset=valset
    acc_train, rmse_train, auc_train = test(net_glob, net_best, dataset_train, dict_users_train, args)
    acc_test, rmse_test, auc_test = test(net_glob, net_best, dataset_test, dict_users_test, args)
    print("Best result at epoch {:3d}".format(epoch_best))
    print("Training accuracy: {:.4f}, Training rmse: {:.4f}, Training auc: {:.4f}".format(acc_train, rmse_train, auc_train))
    print("Testing accuracy: {:.4f}, Testing rmse: {:.4f}, Testing auc: {:.4f}".format(acc_test, rmse_test, auc_test))
    for i in range(school_n):
        dataset_cilent = [stu for stu in dataset_test if stu['school_id'] == i]
        acc_test, rmse_test, auc_test = client_test(net_glob, net_best[i], dataset_cilent, args)
        print("School ID：{:3d}, Testing accuracy: {:.4f}, Testing rmse: {:.4f}, Testing auc: {:.4f}".format(i, acc_test, rmse_test, auc_test))