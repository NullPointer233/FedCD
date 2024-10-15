import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from model import Net
import torch.optim as optim

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        config_file = 'config12.txt'
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        knowledge_dim = int(knowledge_n)
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        log = self.dataset[self.idxs[item]]
        # 一次只读了一个，所以log['knowledge_code']可以执行
        knowledge_emb = [0.] * knowledge_dim
        for knowledge_code in log['knowledge_code']:
            # knowledge_emb[knowledge_code - 1] = 1.0
            knowledge_emb[knowledge_code] = 1.0
        y = log['score']
        # input_stu_ids.append(log['user_id'] - 1)
        # input_exer_ids.append(log['exer_id'] - 1)
        input_stu_ids.append(log['user_id'])
        input_exer_ids.append(log['exer_id'])
        input_knowedge_embs.append(knowledge_emb)
        ys.append(y)

        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowedge_embs), torch.LongTensor(ys)

class LocalUpdate(object):
    def __init__(self, args, lr, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.NLLLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = lr
        a =1
    def train(self, net):
        net.train()
        # train and update


        optimizer = optim.Adam(net.parameters(), self.lr)

        epoch_loss = []

        for iter in range(self.args.local_ep):  # 局部epoch
            batch_loss = []
            for batch_idx, (input_stu_ids, input_exer_ids, input_knowledge_embs, labels) in enumerate(self.ldr_train):
                input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
                    self.args.device), input_exer_ids.to(self.args.device), input_knowledge_embs.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

                output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
                output_0 = torch.ones(output_1.size()).to(self.args.device) - output_1
                output = torch.cat((output_0, output_1), 1)
                loss = self.loss_func(torch.log(output), labels)

                loss.backward()
                optimizer.step()
                net.apply_clipper()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

