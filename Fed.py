import copy
import torch
from torch import nn
"""
str:
'student_emb.weight'
'k_difficulty.weight'
'e_discrimination.weight'
'prednet_full1.weight'
'prednet_full1.bias'
'prednet_full2.weight'
'prednet_full2.bias'
'prednet_full3.weight'
'prednet_full3.bias'
"""
def FedAvg(w, dict_users, school_students):

    dict_num = {}
    for uid in range(len(dict_users)):
        dict_num[uid] = len(dict_users[uid])
    log_sum = sum(dict_num.values())

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k].zero_()

    for k in w_avg.keys():
        if k == 'student_emb.weight':
            for school_id, students in school_students.items():
                for stu in students:
                    w_avg[k][stu] = w[school_id][k][stu]
        elif k == 'k_difficulty.weight' or k == 'e_discrimination.weight':
        # else:
            for i in range(len(w)):
                w_avg[k] += w[i][k] * (dict_num[i] / log_sum)

    for k in w_avg.keys():
        if k == 'student_emb.weight' or k == 'k_difficulty.weight' or k == 'e_discrimination.weight':
            for i in range(len(w)):
                w[i][k] = w_avg[k]

    return w

# replace_keys = {'k_difficulty.weight', 'e_discrimination.weight', 'knowledge_emb', ...}转换为集合