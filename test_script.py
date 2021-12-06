#test script
import pickle
import matplotlib.pyplot as plt
import numpy as np

import torch

import json
# loss_hinge = torch.nn.HingeEmbeddingLoss()

# target = torch.tensor([-1], dtype=torch.float64)
# pre = torch.tensor([1], dtype=torch.float64)


# print(loss_hinge(pre,target))

# pre = torch.tensor([-1], dtype=torch.float64)
# print(loss_hinge(pre,target))

# pre = torch.tensor([0], dtype=torch.float64)
# print(loss_hinge(pre,target))



f = open("BC4GE_data_testsite.json", "r")
data = json.loads(f.read())
f.close()

tp = 0
tn = 0
fp = 0
fn = 0



for prot, prot_scores in data.items():
    score_temp = 0
    score_true = 0
    for id, score_ind in prot_scores.items():
        # print(score_ind["predict:"])
        if score_ind["predict:"]>score_temp:
            score_temp = score_ind["predict:"]
            score_true = score_ind["true:"]

        if score_temp > 0 and score_true > 0:
            tp += 1
        elif score_temp < 0 and score_true > 0:
            fn += 1
        elif score_temp > 0 and score_true == 0:
            fp += 1 
        else:
            tn += 1

print("accuracy:", (tp+tn)/(tp+tn+fp+fn))
print("precision:", tp/(tp+fp))
print("recall:", tp/(tp+fn))
print("F1:", 2*(tp+tn)/(tp+tn+fp+fn)*tp/(tp+fn)/(tp/(tp+fn)+(tp+tn)/(tp+tn+fp+fn)) )
print("tp:", tp)
print("tn:",tn)
print("fp:",fp)
print("fn:",fn)
# tra_error = data['train']
# eval_error = data['eval']

# x = np.arange(30)
# plt.plot(x, tra_error,label="train")
# plt.plot(x, eval_error, label = "vali")
# plt.legend()
# plt.ylim(0, 1)
# plt.xlabel("epoch")
# plt.ylabel("mean BCE-loss")
# plt.show()
# plt.show(block=False)

# f = open("BC4GE_data_evaluation_small.json", "r")
# data = json.loads(f.read())
# f.close()

# tp = 0
# tn = 0
# fp = 0
# fn = 0

# for Num_id, val in data.items():
#     if val["true:"] == 1:
#         if val["predict:"] < 0:
#             tp += 1
#         else: 
#             fp += 1
        
#     elif val["true:"] == -1:
#         if val["loss:"] == 0:
#             tn += 1 
#         else:
#             fn += 1
            
# print("accuracy:", (tp+tn)/(tp+tn+fp+fn))
# print("precision:", tp/(tp+fp))
# print("recall:", tp/(tp+fn))

'''
a_file = open("BC4GE_Score_data_nameVsdefinition.pkl", "rb")
SCORES = pickle.load(a_file)
a_file.close()


a_file = open("BC4GE_Score_name.pkl", "rb")
Prot_names = pickle.load(a_file)
a_file.close()


x = np.arange(25)
y = SCORES[25:]
my_xticks = Prot_names[25:]
plt.xticks(x, my_xticks)
plt.plot(x, y)
plt.ylim(0, 1)
plt.show()
plt.show(block=False)
# input('press <ENTER> to continue')


x = np.arange(25)
y = SCORES[:25]
my_xticks = Prot_names[:25]
plt.xticks(x, my_xticks)
plt.plot(x, y)
plt.ylim(0, 1)
plt.show()
plt.show(block=False)
'''