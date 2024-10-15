import torch
import torch.nn as nn
import copy

class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        #学生嵌入的输入数量为学生数，输出维度是概念的数量
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        #习题对每个概念的考察难度
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        #习题区分学生掌握概念的能力
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        # 上述self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        # 即nn.Embedding(self.emb_num= student_n = 4163, self.stu_dim = self.knowledge_dim = knowledge_n = 123)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        # k_difficulty即nn.Embedding(self.exer_n = 17746, self.knowledge_dim = knowledge_n = 123)
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # 即nn.Embedding(self.exer_n, 1)
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        # 其中prednet_full1为nn.Linear(self.prednet_input_len = self.knowledge_dim = = knowledge_n = 123, self.prednet_len1 = 512)
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        # 即nn.Linear(512, 256)
        output = torch.sigmoid(self.prednet_full3(input_x))
        # 即nn.Linear(256, 1)

        return output

    def apply_clipper(self):
        # 限制三个全连接层的参数为正，进而保证单调假设
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data

    # def setItemCommonality(self, item_commonality):
    #     self.item_commonality = copy.deepcopy(item_commonality)
    #     self.item_commonality.freeze = True


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w)) # 将 w 中的所有值取负再经过一层relu，即a为w中复数的相反数
            w.add_(a)# 即 w = w + a
    #最终效果：参数矩阵负值变为0，正值不变
# if __name__ == '__main__':
#     print(Net(4163,17746,123))