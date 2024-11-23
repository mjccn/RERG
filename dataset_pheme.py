import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
#from transformers import *
import json
from Process import dataset as twitter_Dataset
from torch.utils.data import DataLoader


# global
label2id = {
            "rumor": 0,
            "non-rumor": 1,
            }



class UdGraphDataset(Dataset):
    def __init__(self, fold_x, droprate):

        self.fold_x = fold_x
        #self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]

        # ====================================edgeindex==============================================

        with open('./data/pheme/all/'+ id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        #print(tweets)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        with open('./data/pheme/all/'+ id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]

        #print(inf)
        # id to num
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf
        #print('edgeindex: ', edgeindex.shape)
        #print(id)

        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)

        new_edgeindex = [row, col]
        choose_list = [1, 2]
        probabilities = [0.5, 0.5]
        choose_num1 = twitter_Dataset.random_pick(choose_list, probabilities)
        choose_num2 = twitter_Dataset.random_pick(choose_list, probabilities)

        if self.droprate > 0:
            if choose_num1 == 1:
                # weights = pr_drop_weights(new_edgeindex)  # pagerank
                weights = twitter_Dataset.degree_drop_weights(new_edgeindex)  # degree
                edgeindex_pos1 = twitter_Dataset.drop_edge_weighted(new_edgeindex, weights, 0.4, threshold=0.7)

            elif choose_num1 == 2:
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate))
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row
                edgeindex_pos1 = [row2, col2]

            if choose_num2 == 1:
                # weights = pr_drop_weights(new_edgeindex)  # pagerank
                weights = twitter_Dataset.degree_drop_weights(new_edgeindex)  # degree
                edgeindex_pos2 = twitter_Dataset.drop_edge_weighted(new_edgeindex, weights, 0.4, threshold=0.7)

            elif choose_num2 == 2:
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate))
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row

                edgeindex_pos2 = [row2, col2]

            else:
                edgeindex_pos1 = [row, col]
                edgeindex_pos2 = [row, col]





        #     length = len(row)
        #     poslist = random.sample(range(length), int(length * (1 - self.droprate))) #
        #     poslist = sorted(poslist)
        #     row1 = list(np.array(row)[poslist])
        #     col1 = list(np.array(col)[poslist])
        #
        #     poslist2 = random.sample(range(length), int(length * (1 - self.droprate))) #
        #     poslist2 = sorted(poslist2)
        #     row2 = list(np.array(row)[poslist2])
        #     col2 = list(np.array(col)[poslist2])
        #
        #     new_edgeindex = [row1, col1]
        #     new_edgeindex2 = [row2, col2]
        # else:
        #     new_edgeindex = [row, col]
        #     new_edgeindex2 = [row, col]



        # =========================================X=========================================================
        with open('./bert_w2c/PHEME/pheme_mask/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)

        x_list = json_inf[id]
        x0 = np.array(x_list)

        with open('./data/pheme/pheme_label.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        #y = np.array(y)
        if self.droprate > 0:
            #y = np.array(y)
            zero_list = [0]*768
            x_length = len(x_list)
            r_list = random.sample(range(x_length), int(x_length * self.droprate))
            r_list = sorted(r_list)
            for idex, line in enumerate(x_list):
                for r in r_list:
                    if idex == r:
                        x_list[idex] = zero_list

            x = np.array(x_list)
        else:
            x = np.array(x_list)

        x_orgin = x

        return Data(x0=torch.tensor(x0, dtype=torch.float32),
                    x=torch.tensor(x, dtype=torch.float32),
                    x_orgin=torch.tensor(x_orgin, dtype=torch.float32),
                    edge_index1=torch.LongTensor(edgeindex_pos1),
                    edge_index2=torch.LongTensor(edgeindex_pos2),
                    edgeindex_origin=torch.LongTensor(new_edgeindex),
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y]))



class test_UdGraphDataset(Dataset):
    def __init__(self, fold_x, droprate):


        self.fold_x = fold_x
        #self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]

        # ====================================edgeindex==============================================

        with open('./data/pheme/all/'+ id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        #print(tweets)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        with open('./data/pheme/all/'+ id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]

        #print(inf)
        # id to num
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T

        edgeindex = new_inf
        #print('edgeindex: ', edgeindex.shape)
        #print(id)


        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        #print('new_edgeindex； ', np.array([row, col]).shape)

        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate))) #
            poslist = sorted(poslist)
            row1 = list(np.array(row)[poslist])
            col1 = list(np.array(col)[poslist])

            poslist2 = random.sample(range(length), int(length * (1 - self.droprate))) #
            poslist2 = sorted(poslist2)
            row2 = list(np.array(row)[poslist2])
            col2 = list(np.array(col)[poslist2])

            new_edgeindex = [row1, col1]
            new_edgeindex2 = [row2, col2]
        else:
            new_edgeindex = [row, col]
            new_edgeindex2 = [row, col]



        # =========================================X=========================================================
        with open('./bert_w2c/PHEME/pheme_mask/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)

        x = json_inf[id]
        x = np.array(x)

        with open('./data/pheme/pheme_label.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        #y = np.array(y)

        return Data(x0=torch.tensor(x,dtype=torch.float32),
                x=torch.tensor(x,dtype=torch.float32),
                edge_index1=torch.LongTensor(new_edgeindex),
                edge_index2=torch.LongTensor(new_edgeindex2),
                y1=torch.LongTensor([y]),
                y2=torch.LongTensor([y]),
                id=id)