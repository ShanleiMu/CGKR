# -*- coding: utf-8 -*-
# @Time   : 2021/1/5
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

import torch
import torch.nn.functional as fn

from cgkr.generator import CFGenerator


class CFPosGenerator(CFGenerator):

    def __init__(self, config, dataset, raw_neighbor_relations):
        super(CFPosGenerator, self).__init__(config, dataset, raw_neighbor_relations)

    def generate(self, users, items, kg_neighbors, all_candidates,
                 user_all_embeddings=None, entity_all_embeddings=None,
                 item_embeddings=None):
        neighbors = kg_neighbors[items]  # (batch, n_cans)
        batch_size = users.shape[0]
        batch_tensor = torch.arange(batch_size, device=self.device)
        batch_tensor2 = torch.arange(batch_size * self.replace_num, device=self.device)

        users_e = user_all_embeddings[users]
        items_e = item_embeddings
        ui_e = self.ui_linear(torch.cat([users_e, items_e], dim=1))
        ui_e = fn.normalize(ui_e, p=2, dim=1)
        ui_e = ui_e.unsqueeze(1)    # (batch, 1, embed)

        """Step 1"""
        neighbors_e = self.e_step1_linear(entity_all_embeddings[neighbors])
        neighbors_e = fn.normalize(neighbors_e, p=2, dim=2)  # (batch, n_cans, embed)
        scores = torch.bmm(ui_e, neighbors_e.transpose(1, 2)).squeeze()
        logits = self.step1_softmax(scores)     # (batch, n_cans)
        step1_action_prob, step1_indices = torch.topk(logits, k=self.replace_num, dim=1)    # (batch, num)
        replaced_entities = neighbors[batch_tensor.unsqueeze(1), step1_indices]  # (batch, num)
        selected_relations = self.neighbor_relations[items.unsqueeze(1), step1_indices]  # (batch, num)

        # generate cf_kg_neighbors and action_prob after step1
        blank_entities = torch.zeros((batch_size, self.replace_num), dtype=torch.int64, device=self.device)  # (batch, num)
        cf_kg_neighbors1 = self.get_cf_kg_neighbors(
            kg_neighbors, items, step1_indices, blank_entities)  # (batch, n_cans)
        action_prob1 = torch.mean(step1_action_prob, dim=1)
        action_prob1 = torch.log(action_prob1)  # (batch)

        """Step 2"""
        # filtered by score between user and entity
        candidates = all_candidates[selected_relations]     # (batch, num, n_cans)
        candidates_e = entity_all_embeddings[candidates]
        candidates_e = candidates_e.view(batch_size, self.replace_num * self.n_cans, self.embedding_size)   # (batch, num * n_cans, embed)
        scores = torch.bmm(users_e.unsqueeze(1), candidates_e.transpose(1, 2)).squeeze()
        scores = scores.view(batch_size, self.replace_num, self.n_cans)   # (batch, num, n_cans)
        _, remain_indices = torch.topk(scores, k=self.remain_cans, dim=2)  # (batch, num, remain_cans)
        candidates = candidates.view(batch_size * self.replace_num, self.n_cans)
        remain_indices = remain_indices.view(batch_size * self.replace_num, self.remain_cans)
        candidates = candidates[batch_tensor2.unsqueeze(1), remain_indices]  # (batch * num ,remain_cans)

        # sampled by policy
        replaced_e = self.e_step1_linear(entity_all_embeddings[replaced_entities])
        replaced_e = fn.normalize(replaced_e, p=2, dim=2)  # (batch, num, embed)
        replaced_e = replaced_e.view(batch_size * self.replace_num, self.embedding_size)
        candidates_e = self.e_step2_linear(entity_all_embeddings[candidates])
        candidates_e = fn.normalize(candidates_e, p=2, dim=2)  # (batch * num, remain_cans, embed)
        ui_e = ui_e.repeat(1, self.replace_num, 1).view(batch_size * self.replace_num, self.embedding_size)
        scores = torch.bmm((ui_e * replaced_e).unsqueeze(1), candidates_e.transpose(1, 2)).squeeze()
        logits = self.step2_softmax(scores)     # (batch * num, remain_cans)

        step2_indices = torch.multinomial(logits, 1).squeeze(1)  # (batch * num)
        step2_action_prob = logits[batch_tensor2, step2_indices]
        step2_action_prob = step2_action_prob.view(batch_size, self.replace_num)
        selected_entities = candidates[batch_tensor2, step2_indices]
        selected_entities = selected_entities.view(batch_size, self.replace_num)

        # generate cf_kg_neighbors and action_prob after step2
        cf_kg_neighbors2 = self.get_cf_kg_neighbors(kg_neighbors, items,
                                                    step1_indices, selected_entities)
        action_prob2 = torch.log(torch.mean(step1_action_prob, dim=1)) + \
            torch.log(torch.mean(step2_action_prob, dim=1))

        return cf_kg_neighbors1, action_prob1, cf_kg_neighbors2, action_prob2
