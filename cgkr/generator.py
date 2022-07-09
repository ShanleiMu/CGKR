# -*- coding: utf-8 -*-
# @Time   : 2020/12/28
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization


class CFGenerator(nn.Module):

    def __init__(self, config, dataset, raw_neighbor_relations):
        super(CFGenerator, self).__init__()

        # load parameters info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ENTITY_ID = config['ENTITY_ID_FIELD']
        self.device = config['device']

        self.embedding_size = config['embedding_size']
        self.n_cans = config['n_cans']
        self.remain_cans = config['remain_cans']
        self.replace_num = config['replace_num']

        # load dataset info
        self.n_items = dataset.num(self.ITEM_ID)
        self.neighbor_relations = \
            torch.from_numpy(raw_neighbor_relations).to(self.device)

        # define layers
        self.ui_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.e_step1_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.e_step2_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.step1_softmax = nn.Softmax(dim=1)
        self.step2_softmax = nn.Softmax(dim=1)

        self.apply(xavier_normal_initialization)

    def generate(self, users, items, kg_neighbors, all_candidates,
                 user_all_embeddings=None, entity_all_embeddings=None, item_embeddings=None):
        raise NotImplementedError()

    @staticmethod
    def get_cf_kg_neighbors(kg_neighbors, batch_tensor, indices, values):
        cf_kg_neighbors = kg_neighbors.clone()
        cf_kg_neighbors[batch_tensor.unsqueeze(1), indices] = values
        return cf_kg_neighbors
