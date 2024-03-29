import os
import pickle
import sys
import copy
import logging
import random
import time
from collections import defaultdict
from os.path import join,dirname
import multiprocessing
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from whoiswho import logger
from whoiswho.dataset import load_utils
from whoiswho.config import RNDFilePathConfig, log_time, version2path
from whoiswho.utils import set_log, load_pickle, save_pickle, load_json, save_json
from whoiswho.loadmodel.ClassficationModels import GBDTModel,FeatDataLoader

def get_cell_pred(cell_model, unass_pid2aid, eval_feat_data, cell_feat_list):
    unass_pid2aid2score = defaultdict(dict)
    for unass_pid, candi_aids in tqdm(unass_pid2aid):
        candi_feat = []
        for candi_aid in candi_aids:
            candi_feat.append(eval_feat_data.get_whole_feat(unass_pid, candi_aid, cell_feat_list))
        candi_preds = cell_model.predict(candi_feat)
        for candi_index, candi_aid in enumerate(candi_aids):
            unass_pid2aid2score[unass_pid][candi_aid] = float(candi_preds[candi_index])
    return unass_pid2aid2score


def get_result(cell_model, unass_pid2aid, eval_feat_data, cell_config, cell_i, res_unass_aid2score_list,
               cell_weight_sum, model_save_dir, info):
    this_cell_res = defaultdict(dict)
    unass_pid2aid_cell_i = get_cell_pred(cell_model, unass_pid2aid, eval_feat_data, cell_config['feature_list'])

    for unass_pid, unass_aids in unass_pid2aid:
        for unass_aid in unass_aids:
            this_cell_res[unass_pid][unass_aid] = unass_pid2aid_cell_i[unass_pid][unass_aid]
            if cell_i == 0:
                res_unass_aid2score_list[unass_pid][unass_aid] = unass_pid2aid_cell_i[unass_pid][unass_aid] * \
                                                                 cell_config['cell_weight'] / cell_weight_sum
            else:
                res_unass_aid2score_list[unass_pid][unass_aid] += unass_pid2aid_cell_i[unass_pid][unass_aid] * \
                                                                  cell_config['cell_weight'] / cell_weight_sum
    save_json(dict(this_cell_res), model_save_dir, f'result_score_cell{cell_i}.{info}.json')


def deal_nil_threshold_new(score_path, save_dir, info, thres=0.7):
    res_unass_aid2score_list = load_json(score_path)
    result = defaultdict(list)
    ass_papers = 0
    max_score = -1
    for pid, aid2score in res_unass_aid2score_list.items():
        tmp_scores = []
        for aid, score in aid2score.items():
            tmp_scores.append((aid, score))
        if len(tmp_scores) == 0:
            continue
        tmp_scores.sort(key=lambda x: x[1], reverse=True)
        max_score = max(max_score, tmp_scores[0][1])
        if tmp_scores[0][1] >= thres:
            ass_papers += 1
            result[tmp_scores[0][0]].append(pid.split('-')[0])
    log_msg = f'ass_papers= {ass_papers}, max_score={max_score}.'
    print(log_msg)
    logger.warning(log_msg)
    save_json(dict(result), save_dir, f'result.{info}.json')



def test_config2data(test_config,debug_mod=False):
    eval_feat_data = FeatDataLoader(test_config)
    unass_list = load_json(test_config['unass_path'])
    unass_name2aid2pid_v1 = load_json(test_config['name2aid2pid'])
    unass_pid2aid = []  # [pid, [candi_aid,]]

    if debug_mod:
        unass_list = unass_list[:20]

    for unass_pid, name in unass_list:
        # candi_aids = list(unass_name2aid2pid_v1[name])
        candi_aids = [aid for aid in unass_name2aid2pid_v1[name] if unass_name2aid2pid_v1[name][aid]]
        unass_pid2aid.append((unass_pid, candi_aids))
    return eval_feat_data, unass_pid2aid


class RNDTrainer:
    def __init__(self, version, debug=False , datatype = None , raw_data_root = None,processed_data_root = None, hand_feat_root=None, bert_feat_root=None,
                 simplified = False, graph_data=False):
        self.v2path = version2path(version)
        self.name = self.v2path['name']
        self.task = self.v2path['task']  # RND SND
        assert self.task == 'RND', 'This features' \
                                   'only support RND task'

        self.debug = debug

        # Modifying arguments when calling from outside
        self.type = datatype
        self.raw_data_root = raw_data_root
        self.processed_data_root = processed_data_root
        self.hand_feat_root = hand_feat_root
        self.bert_feat_root = bert_feat_root
        self.graph_data = graph_data  #Whether use the whoiswhograph data

        if not datatype:
            self.type = self.v2path['type']  # train valid test
        if not raw_data_root:
            self.raw_data_root = self.v2path['raw_data_root']
        if not processed_data_root:
            self.processed_data_root = self.v2path['processed_data_root']
        if not hand_feat_root:
            self.hand_feat_root = self.v2path['hand_feat_root']
        if not bert_feat_root:
            self.bert_feat_root = self.v2path['bert_feat_root']

        # print(f'processed_data_root: {self.processed_data_root}\nhand_feat_root: {self.hand_feat_root}\nbert_feat_root: {self.bert_feat_root}\n ')

        # The train data configuration, which is related to create GBDT models
        self.train_config_list = [
        {
            'train_path': self.processed_data_root + "/train/kfold_dataset/kfold_v1/train_ins.json",
            'dev_path'  : self.processed_data_root + "/train/kfold_dataset/kfold_v1/test_ins.json",
        },
        {
            'train_path': self.processed_data_root + "/train/kfold_dataset/kfold_v2/train_ins.json",
            'dev_path'  : self.processed_data_root + "/train/kfold_dataset/kfold_v2/test_ins.json",
        },
        {
            'train_path': self.processed_data_root + "/train/kfold_dataset/kfold_v3/train_ins.json",
            'dev_path'  : self.processed_data_root + "/train/kfold_dataset/kfold_v3/test_ins.json",
        },
        {
            'train_path': self.processed_data_root + "/train/kfold_dataset/kfold_v4/train_ins.json",
            'dev_path'  : self.processed_data_root + "/train/kfold_dataset/kfold_v4/test_ins.json",
        },
        {
            'train_path': self.processed_data_root + "/train/kfold_dataset/kfold_v5/train_ins.json",
            'dev_path'  : self.processed_data_root + "/train/kfold_dataset/kfold_v5/test_ins.json",
        },
        ]

        # train feature
        self.train_feature_config = {
            'bert_path': self.bert_feat_root + 'pid2aid2bert_feat.offline.pkl',
            'hand_path': self.hand_feat_root + 'pid2aid2hand_feat.offline.pkl',

        }
        # valid set and test set config
        self.test_config_v1 = {
            'bert_path': self.bert_feat_root + 'pid2aid2bert_feat.onlinev1.pkl',
            'hand_path': self.hand_feat_root + 'pid2aid2hand_feat.onlinev1.pkl',
            'unass_path': self.processed_data_root + RNDFilePathConfig.unass_candi_v1_path,
            'name2aid2pid': self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
        }

        self.test_config_v2 = {
            'bert_path': self.bert_feat_root + 'pid2aid2bert_feat.onlinev2.pkl',
            'hand_path': self.hand_feat_root + 'pid2aid2hand_feat.onlinev2.pkl',
            'unass_path': self.processed_data_root + RNDFilePathConfig.unass_candi_v2_path,
            'name2aid2pid': self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
        }


        if self.graph_data:
            self.whoiswhograph_extend_processed_data = self.v2path['whoiswhograph_extend_processed_data']
            self.graph_feat_root = self.v2path['graph_feat_root']
            self.graph_train_config_list = [
                {
                    'train_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v1/train_ins.json",
                    'dev_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v1/test_ins.json",
                },
                {
                    'train_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v2/train_ins.json",
                    'dev_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v2/test_ins.json",
                },
                {
                    'train_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v3/train_ins.json",
                    'dev_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v3/test_ins.json",
                },
                {
                    'train_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v4/train_ins.json",
                    'dev_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v4/test_ins.json",
                },
                {
                    'train_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v5/train_ins.json",
                    'dev_path': self.whoiswhograph_extend_processed_data + "/train/kfold_dataset/kfold_v5/test_ins.json",
                },
            ]

            # Check if graph_data related features are stored.
            if not os.path.exists(self.hand_feat_root + 'whoiswhograph_pid2aid2hand_feat.offline.pkl'):
                print("Please Generating Graph Feature First!")
                raise RuntimeError

            # training set feature
            self.graph_train_feature_config = {
                'hand_path': self.hand_feat_root + 'whoiswhograph_pid2aid2hand_feat.offline.pkl',
                'graph_path': self.graph_feat_root + 'pid2aid2graph_feat_gat.offline.pkl',
            }
            # valid set and test set config
            self.graph_test_config_v1 = {
                'hand_path': self.hand_feat_root + 'pid2aid2hand_feat.onlinev1.pkl',
                'graph_path': self.graph_feat_root + 'pid2aid2graph_feat_gat.onlinev1.pkl',
                'unass_path': self.whoiswhograph_extend_processed_data + RNDFilePathConfig.unass_candi_v1_path,
                'name2aid2pid': self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
            }
            self.graph_test_config_v2 = {
                'hand_path': self.hand_feat_root + 'pid2aid2hand_feat.onlinev2.pkl',
                'graph_path': self.graph_feat_root + 'pid2aid2graph_feat_gat.onlinev2.pkl',
                'unass_path': self.whoiswhograph_extend_processed_data + RNDFilePathConfig.unass_candi_v2_path,
                'name2aid2pid': self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
            }


        # model_save_dir
        self.model_save_dir = join(os.path.abspath(dirname(__file__)), f'{self.task}_save_model', '')
        os.makedirs(self.model_save_dir, exist_ok=True)
        if self.graph_data:
            self.graph_model_save_dir = join(os.path.abspath(dirname(__file__)), f'{self.task}_graph_save_model','')
            os.makedirs(self.graph_model_save_dir, exist_ok=True)

        # result_save_dir
        self.result_save_dir = join(os.path.abspath(dirname(__file__)), f'{self.task}_result', '')
        os.makedirs(self.result_save_dir, exist_ok=True)
        # if self.graph_data:
        #     self.graph_result_save_dir = join(os.path.abspath(dirname(__file__)), f'{self.task}_graph_result','')
        #     os.makedirs(self.graph_result_save_dir, exist_ok=True)


        self.semantic_model = GBDTModel(self.train_config_list,
                               os.path.join(self.model_save_dir,f'{log_time}'),
                               simplified = simplified,  # use simplified gbdt model, only one catboost
                               debug = self.debug)

        if self.graph_data:
            self.relational_model = GBDTModel(self.graph_train_config_list,
                                            os.path.join(self.graph_model_save_dir, f'{log_time}'),
                                            graph_data=graph_data,
                                            debug=self.debug)
    # use train data
    def fit(self):
        # train semantic & relational models
        if self.graph_data:
            # train_feature_config
            cell_model_list = self.semantic_model.fit(self.train_feature_config)
            # graph_train_feature_config
            graph_cell_model_list = self.relational_model.fit(self.graph_train_feature_config)
            return (cell_model_list, graph_cell_model_list)
        # train semantic model
        else:
            cell_model_list = self.semantic_model.fit(self.train_feature_config)
            return (cell_model_list,None)

    def predict(self,whole_cell_model_list = None, cell_model_path_list: List[str] = None,
                graph_cell_model_list = None, graph_cell_model_path_list: List[str] = None, datatype = None):
        try:
            cell_model_list = whole_cell_model_list[0]
        except:
            pass

        #If datatype is entered in predict, change self.datatype
        if datatype:
            self.type = datatype

        if self.type == 'valid':
            test_config = self.test_config_v1
            eval_feat_data, unass_pid2aid = test_config2data(test_config,debug_mod=self.debug)
        else: #test
            test_config = self.test_config_v2
            eval_feat_data, unass_pid2aid = test_config2data(test_config,debug_mod=self.debug)

        if cell_model_path_list:
            cell_model_list = self.semantic_model.load(cell_model_path_list)
        logger.info('predicting...')
        res_unass_aid2score_list = defaultdict(dict)
        for cell_i, cell_config in enumerate(self.semantic_model.cell_list_config):
            cell_model = cell_model_list[cell_i]
            eval_feat_data.update_feat(cell_config['feature_list'])
            get_result(cell_model, unass_pid2aid, eval_feat_data, cell_config, cell_i,
                       res_unass_aid2score_list,
                       self.semantic_model.cell_weight_sum,
                       self.result_save_dir,
                       f'{self.type}')

        if self.graph_data :
            try:
                graph_cell_model_list = whole_cell_model_list[1]
            except:
                pass
            if self.type == 'valid':
                test_config = self.graph_test_config_v1
                eval_feat_data, unass_pid2aid = test_config2data(test_config, debug_mod=self.debug)
            else:  # test
                test_config = self.graph_test_config_v2
                eval_feat_data, unass_pid2aid = test_config2data(test_config, debug_mod=self.debug)

            if graph_cell_model_path_list:
                graph_cell_model_list = self.relational_model.load(graph_cell_model_path_list)

            logger.info('predicting...')
            for cell_i, cell_config in enumerate(self.relational_model.cell_list_config):
                cell_model = graph_cell_model_list[cell_i]
                eval_feat_data.update_feat(cell_config['feature_list'])
                get_result(cell_model, unass_pid2aid, eval_feat_data, cell_config,
                           cell_i + len(self.semantic_model.cell_list_config), #cell number
                           res_unass_aid2score_list,
                           15,
                           self.result_save_dir,
                           f'{self.type}')

        # Store the final results of voting by all cell_models
        score_result_path = os.path.join(self.result_save_dir, f'result_score_vote.{self.type}.json')
        save_json(dict(res_unass_aid2score_list), score_result_path)

        # Set a threshold to filter NIL
        deal_nil_threshold_new(
            score_result_path, self.result_save_dir, f'{self.type}', 0.65

        )

if __name__ == '__main__':
    version = {"name": 'v3', "task": 'RND', "type": 'train'}
    trainer = RNDTrainer(version,graph_data=True,simplified=True)
    cell_model_list = trainer.fit()
    logger.info("Finish Train data")

    version = {"name": 'v3', "task": 'RND', "type": 'valid'}
    trainer = RNDTrainer(version,graph_data=True,simplified=True)
    trainer.predict(whole_cell_model_list=cell_model_list)
    logger.info("Finish Valid data")

    version = {"name": 'v3', "task": 'RND', "type": 'test'}
    trainer = RNDTrainer(version,graph_data=True,simplified=True)
    trainer.predict(whole_cell_model_list=cell_model_list)
    logger.info("Finish Test data")



