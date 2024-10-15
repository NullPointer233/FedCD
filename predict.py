import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Update import DatasetSplit


def test(net_g, w_locals, dataset, dict_users, args):

    idxs_users = np.array(range(args.num_clients))
    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []

    for idx in idxs_users:
        net_g.load_state_dict(w_locals[idx])
        net_g.eval()
        data_loader = DataLoader(DatasetSplit(dataset, dict_users[idx]), batch_size=args.bs)

        for idxxx, (input_stu_ids, input_exer_ids, input_knowledge_embs, labels) in enumerate(data_loader):
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
                args.device), input_exer_ids.to(args.device), input_knowledge_embs.to(args.device), labels.to(
                args.device)
            output = net_g(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output = output.view(-1)
            labels = labels.view(-1)
            for i in range(len(labels)):
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            pred_all += output.tolist()
            label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (args.epochs, accuracy, rmse, auc))

    return accuracy, rmse, auc

def client_test(net_g, net_dict, dataset, args):
    net_g.load_state_dict(net_dict)
    net_g.eval()
    # testing
    all_idxs = [i for i in range(len(dataset))]
    data_loader = DataLoader(DatasetSplit(dataset, all_idxs), batch_size=args.bs)
    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    for idx, (input_stu_ids, input_exer_ids, input_knowledge_embs, labels) in enumerate(data_loader):
        # if args.gpu != -1:
        #     data, target = data.cuda(), target.cuda()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(args.device), input_exer_ids.to(args.device), input_knowledge_embs.to(args.device), labels.to(args.device)
        output = net_g(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        labels = labels.view(-1)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (args.epochs, accuracy, rmse, auc))

    return accuracy, rmse, auc