# -*- coding: utf-8 -*-
# @Time   : 2020/12/28
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

import torch
import torch.optim as optim
import numpy as np

from recbole.trainer import Trainer
from recbole.utils import KGDataLoaderState


class CFTrainer(Trainer):

    def __init__(self, config, dataset, rec_model, raw_kg_neighbors,
                 cf_pos_generator=None, cf_neg_generator=None):
        super(CFTrainer, self).__init__(config, rec_model)

        # load parameters info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ENTITY_ID = config['ENTITY_ID_FIELD']
        self.RELATION_ID = config['RELATION_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.device = config['device']

        self.glr = config['glr']
        self.gamma = config['gamma']
        self.cf_pos_flag = config['cf_pos_flag']
        self.cf_neg_flag = config['cf_neg_flag']
        self.max_neighbor_size = config['max_neighbor_size']
        self.n_cans = config['n_cans']
        self.replace_step = config['replace_step']
        self.train_recommender = config['train_recommender']
        self.train_generator = config['train_generator']

        # load dataset info
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID)
        self.kg_neighbors = torch.from_numpy(raw_kg_neighbors).to(self.device)
        self.r2candidates = dataset.relation2candidates()

        # init generator and optimizer
        if self.cf_pos_flag:
            self.cf_pos_generator = cf_pos_generator
            self.cf_pos_optimizer = optim.Adam(self.cf_pos_generator.parameters(),
                                               lr=self.glr)
        if self.cf_neg_flag:
            self.cf_neg_generator = cf_neg_generator
            self.cf_neg_optimizer = optim.Adam(self.cf_neg_generator.parameters(),
                                               lr=self.glr)

    def generate_cf_kg(self, interaction, user_all_embeddings, entity_all_embeddings,
                       flag='pos'):
        if flag == 'pos':
            items = interaction[self.ITEM_ID]
            cf_generator = self.cf_pos_generator
        else:
            items = interaction[self.NEG_ITEM_ID]
            cf_generator = self.cf_neg_generator
        users = interaction[self.USER_ID]
        kg_neighbors = self.kg_neighbors.clone()
        for i in range(self.replace_step):
            item_embeddings = self.model.get_cf_i_embeddings(items, kg_neighbors)
            _, _, kg_neighbors, _ = cf_generator.generate(
                users, items, kg_neighbors, self.sub_candidates,
                user_all_embeddings, entity_all_embeddings, item_embeddings)
        return kg_neighbors

    def calculate_generator_loss(self, interaction, user_all_embeddings,
                                 entity_all_embeddings, flag='pos'):
        if flag == 'pos':
            items = interaction[self.ITEM_ID]
            cf_generator = self.cf_pos_generator
            generate_reward = self.model.generate_pos_reward
        else:
            items = interaction[self.NEG_ITEM_ID]
            cf_generator = self.cf_neg_generator
            generate_reward = self.model.generate_neg_reward
        users = interaction[self.USER_ID]
        kg_neighbors = self.kg_neighbors.clone()
        batch_g_loss = None
        for i in range(self.replace_step):
            with torch.no_grad():
                item_embeddings = self.model.get_cf_i_embeddings(items, kg_neighbors)
            kg_neighbors_prev, action_prob1, kg_neighbors, action_prob2 = \
                cf_generator.generate(
                    users, items, kg_neighbors, self.sub_candidates,
                    user_all_embeddings, entity_all_embeddings, item_embeddings)
            with torch.no_grad():
                step_reward1, step_reward2 = generate_reward(
                    interaction, kg_neighbors_prev, kg_neighbors,
                    user_all_embeddings, entity_all_embeddings)

            step_loss = - torch.mean(step_reward1 * action_prob1)
            step_loss -= torch.mean(step_reward2 * action_prob2)
            step_loss = self.gamma ** i * step_loss
            batch_g_loss = step_loss if batch_g_loss is None \
                else batch_g_loss + step_loss
        batch_g_loss /= self.replace_step

        return batch_g_loss

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        total_loss = list()
        total_base_loss, total_pos_g_loss, total_neg_g_loss = None, None, None
        self.sub_candidates = build_sub_candidates(self.r2candidates, self.n_items,
                                                   self.n_entities, self.n_relations,
                                                   self.n_cans).to(self.device)
        interaction_state = KGDataLoaderState.RS
        train_data.set_mode(interaction_state)

        """ ----- Train Recommender ----- """
        if self.train_recommender:
            self.model.train()
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                user_all_embeddings, entity_all_embeddings = self.model.forward()
                with torch.no_grad():
                    # Generate cf kg
                    if self.cf_pos_flag:
                        kg_neighbors = self.generate_cf_kg(
                            interaction, user_all_embeddings, entity_all_embeddings,
                            flag='pos')
                        interaction.interaction['cf_pos_kg_neighbors'] = kg_neighbors
                    if self.cf_neg_flag:
                        kg_neighbors = self.generate_cf_kg(
                            interaction, user_all_embeddings, entity_all_embeddings,
                            flag='neg')
                        interaction.interaction['cf_neg_kg_neighbors'] = kg_neighbors
                losses = self.model.calculate_loss(interaction, user_all_embeddings,
                                                   entity_all_embeddings)
                loss = sum(losses)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_base_loss = loss_tuple if total_base_loss is None else \
                    tuple(map(sum, zip(total_base_loss, loss_tuple)))

        """ ----- Train Generator ----- """
        if self.train_generator:
            with torch.no_grad():
                user_all_embeddings, entity_all_embeddings = self.model.forward()

            if self.cf_pos_flag:
                self.cf_pos_generator.train()
                for batch_idx, interaction in enumerate(train_data):
                    interaction = interaction.to(self.device)
                    self.cf_pos_optimizer.zero_grad()
                    batch_pos_g_loss = self.calculate_generator_loss(
                        interaction, user_all_embeddings, entity_all_embeddings,
                        flag='pos')
                    self._check_nan(batch_pos_g_loss)
                    batch_pos_g_loss.backward()
                    self.cf_pos_optimizer.step()
                    total_pos_g_loss = batch_pos_g_loss if total_pos_g_loss is None \
                        else total_pos_g_loss + batch_pos_g_loss

            if self.cf_neg_flag:
                self.cf_neg_generator.train()
                for batch_idx, interaction in enumerate(train_data):
                    interaction = interaction.to(self.device)
                    self.cf_neg_optimizer.zero_grad()
                    batch_neg_g_loss = self.calculate_generator_loss(
                        interaction, user_all_embeddings, entity_all_embeddings,
                        flag='neg')
                    self._check_nan(batch_neg_g_loss)
                    batch_neg_g_loss.backward()
                    self.cf_neg_optimizer.step()
                    total_neg_g_loss = batch_neg_g_loss if total_neg_g_loss is None \
                        else total_neg_g_loss + batch_neg_g_loss

        """Return loss"""
        if self.train_recommender:
            for loss in total_base_loss:
                total_loss.append(loss)
        if self.train_generator:
            if self.cf_pos_flag:
                total_loss.append(total_pos_g_loss)
            if self.cf_neg_flag:
                total_loss.append(total_neg_g_loss)

        return tuple(total_loss)


def build_sub_candidates(relation2candidates, n_items, n_entities, n_relations, threshold):
    candidates = []
    for i in range(n_relations):
        if i in relation2candidates:
            candidates_length = len(relation2candidates[i])
            if candidates_length > threshold:
                candidate_entities = np.random.choice(
                    relation2candidates[i], size=threshold, replace=False)
                candidates.append(candidate_entities)
            else:
                candidate_entities = \
                    np.random.choice(relation2candidates[i],
                                     size=threshold - candidates_length, replace=True)
                candidate_entities = np.hstack((relation2candidates[i], candidate_entities))
                candidates.append(candidate_entities)
        else:
            candidate_entities = np.random.choice(range(n_items, n_entities),
                                                  size=threshold, replace=False)
            candidates.append(candidate_entities)
    return torch.from_numpy(np.array(candidates))
