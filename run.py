# -*- coding: utf-8 -*-
# @Time   : 2020/12/28
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

import argparse

from logging import getLogger

from cgkr.model import CGKR
from cgkr.cf_pos_generator import CFPosGenerator
from cgkr.cf_neg_generator import CFNegGenerator
from cgkr.trainer import CFTrainer

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger


def objective_function(config_dict=None, config_file_list=None, saved=True):

    config = Config(model=CGKR,
                    config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering and splitting
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    raw_kg_neighbors, raw_neighbor_relations, _ = \
        train_data.kg_neighbors(config['max_neighbor_size'],
                                relations=True, random=True, repeat=True)

    # model loading and initialization
    cf_pos_generator, cf_neg_generator = None, None
    rec_model = CGKR(config, train_data, raw_kg_neighbors).to(config['device'])
    logger.info(rec_model)
    if config['cf_pos_flag']:
        cf_pos_generator = CFPosGenerator(config, train_data,
                                          raw_neighbor_relations).to(config['device'])
        logger.info(cf_pos_generator)
    if config['cf_neg_flag']:
        cf_neg_generator = CFNegGenerator(config, train_data,
                                          raw_neighbor_relations).to(config['device'])
        logger.info(cf_neg_generator)

    # trainer initialization
    trainer = CFTrainer(config, train_data, rec_model, raw_kg_neighbors,
                        cf_pos_generator, cf_neg_generator)
    if config['pretrained_model_path']:
        trainer.resume_checkpoint(config['pretrained_model_path'])

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=True, saved=saved)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved)
    print(test_result)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='yaml/overall.yaml')
    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(',')
    objective_function(config_file_list=config_file_list, config_dict=None,
                       saved=True)
