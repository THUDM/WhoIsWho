import os
import pickle
import sys
import copy
import logging
import random
import time
from collections import defaultdict
import multiprocessing

import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
sys.path.append('../../')
from whoiswho.config import RNDFilePathConfig, log_time, version2path
from whoiswho.utils import set_log, load_pickle, save_pickle, load_json, save_json
from whoiswho import logger
xgb_njob = int(multiprocessing.cpu_count() / 2)


def random_select_instance(train_ins, nil_ratio=0.2, min_neg_num=10):
    train_ins_num = len(train_ins)
    res_list = []
    nil_ins_max = int(train_ins_num * nil_ratio)
    random.shuffle(train_ins)
    neg_ins = train_ins[0:nil_ins_max]
    for ins in neg_ins:
        if len(ins[3]) >= min_neg_num:
            ins[2] = ''
            res_list.append(ins)
    pos_ins = train_ins[nil_ins_max:]
    for ins in pos_ins:
        if len(ins[3]) >= min_neg_num:
            ins[3] = random.sample(ins[3], len(ins[3]) - 1)
            res_list.append(ins)
    random.shuffle(res_list)
    return res_list


def get_gbd_model(gbd_type='xgb', njob=xgb_njob, model_args=None):
    '''

    Args:
        gbd_type: ['xgb', 'cat', 'lgbm']
        njob: multiprocess
        model_args:

    Returns:

    '''
    assert gbd_type in ['xgb', 'cat', 'lgbm']
    gbd_model = None
    if gbd_type == 'xgb':
        default_config = {
            'max_depth'       : 7,
            'learning_rate'   : 0.01,
            'n_estimators'    : 1000,
            'subsample'       : 0.8,
            'n_jobs'          : njob,
            'min_child_weight': 6,
            'random_state'    : 666
        }
        if model_args is not None:
            default_config.update(model_args)
        gbd_model = XGBClassifier(**default_config)
    elif gbd_type == 'cat':
        default_config = {
            'iterations'   : 1000,
            'learning_rate': 0.05,
            'depth'        : 10,
            'loss_function': 'Logloss',
            'eval_metric'  : 'Logloss',
            'random_seed'  : 666,
        }
        if model_args is not None:
            default_config.update(model_args)
        gbd_model = CatBoostClassifier(**default_config)
    elif gbd_type == 'lgbm':
        default_config = {
            'max_depth'    : 10,
            'learning_rate': 0.01,
            'n_estimators' : 1000,
            'objective'    : 'binary',
            'subsample'    : 0.8,
            'n_jobs'       : njob,
            'num_leaves'   : 82,
            'random_state' : 666
        }
        if model_args is not None:
            default_config.update(model_args)
        gbd_model = LGBMClassifier(**default_config)
    return gbd_model


def get_gbd_pred(gbd_model, feat, gbd_type='xgb'):
    assert gbd_type in ['xgb', 'cat', 'lgbm']
    if gbd_type in ['xgb']:
        feat = np.array(feat)
        res = gbd_model.predict_proba(feat)[:, 1]
    elif gbd_type in ['lgbm']:
        feat = np.array(feat)
        res = gbd_model.predict_proba(feat)[:, 1]
    elif gbd_type in ['cat']:
        res = gbd_model.predict_proba(feat)[:, 1]
    else:
        raise NotImplementedError
    return res


def fit_gbd_model(gbd_model, whole_x, whole_y, gbd_type='xgb'):
    assert gbd_type in ['xgb', 'cat', 'lgbm']
    if gbd_type in ['xgb', 'cat', 'lgbm']:
        gbd_model.fit(whole_x, whole_y)
    else:
        raise NotImplementedError



class FeatDataLoader:
    def __init__(self, feat_config):
        self.bert_path = feat_config['bert_path']
        self.hand_dict = load_pickle(feat_config['hand_path'])
        self.bert_dict = None

        self.graph_path = feat_config['graph_path']
        self.graph_dict = None

    def update_feat(self, feat_list):
        if 'bert' in feat_list and self.bert_dict is None:
            self.bert_dict = load_pickle(self.bert_path)
        if 'graph' in feat_list and self.graph_dict is None:
            self.graph_dict = load_pickle(self.graph_path)

    def get_whole_feat(self, unass_pid, candi_aid, feat_list):
        hand_feat = self.hand_dict[unass_pid][candi_aid]
        if 'bert' in feat_list:
            whole_feature = np.hstack([self.bert_dict[unass_pid][candi_aid], hand_feat])
        else:
            whole_feature = np.array(hand_feat)

        if 'graph' in feat_list:
            whole_feature = np.hstack([self.graph_dict[unass_pid][candi_aid],whole_feature])

        return whole_feature


class CellModel:
    def __init__(self, model_config, kfold, debug_mod = False):
        # print(model_config)
        assert len(model_config) <= 2
        lv1_model_list = []
        lv2_model_list = []
        lv1_gdb_type = []
        lv2_gdb_type = []
        for i in range(kfold):
            this_fold_model_list = []
            for t_config in model_config[0]:
                if 'params' in t_config and len(t_config['params'].keys()) > 0:
                    this_fold_model_list.append(
                        get_gbd_model(t_config['gbd_type'], njob=xgb_njob, model_args=t_config['params']))
                else:
                    this_fold_model_list.append(get_gbd_model(t_config['gbd_type'], njob=xgb_njob))
                if i == 0:
                    lv1_gdb_type.append(t_config['gbd_type'])
            lv1_model_list.append(this_fold_model_list)
        if len(model_config) == 2 and len(model_config[1]) > 0:
            for t_config in model_config[1]:
                if 'params' in t_config and len(t_config['params'].keys()) > 0:
                    lv2_model_list.append(
                        get_gbd_model(t_config['gbd_type'], njob=xgb_njob, model_args=t_config['params']))
                else:
                    lv2_model_list.append(get_gbd_model(t_config['gbd_type'], njob=xgb_njob))
                lv2_gdb_type.append(t_config['gbd_type'])
        self.lv1_model_list = lv1_model_list
        self.lv2_model_list = lv2_model_list
        self.lv1_gdb_type = lv1_gdb_type
        self.lv2_gdb_type = lv2_gdb_type
        self.kfold = kfold
        self.has_lv2 = True if len(lv2_gdb_type) > 0 else False

        self.debug_mod = debug_mod

    def fit(self, whole_x, whole_y, training_type, fold_i=None):
        print('\tfitting ', training_type)
        print('\t\twhole_x.shape ', whole_x.shape)
        print('\t\twhole_y.shape ', whole_y.shape)
        assert training_type == 'lv2' or fold_i is not None
        if training_type == 'lv1':
            for lv1_model, gdb_type in zip(self.lv1_model_list[fold_i], self.lv1_gdb_type):
                fit_gbd_model(lv1_model, whole_x, whole_y, gdb_type)
            return
        elif training_type == 'lv2':
            assert self.has_lv2
            for lv2_model, gdb_type in zip(self.lv2_model_list, self.lv2_gdb_type):
                fit_gbd_model(lv2_model, whole_x, whole_y, gdb_type)
            return
        raise ValueError('illegal training_type ', training_type)

    def _get_lv1_preds(self, candis_feature, fold_i):
        preds = None
        for lv1_model, gdb_type in zip(self.lv1_model_list[fold_i], self.lv1_gdb_type):
            if preds is None:
                preds = get_gbd_pred(lv1_model, candis_feature, gdb_type)[:, np.newaxis]
            else:
                preds = np.hstack([preds, get_gbd_pred(lv1_model, candis_feature, gdb_type)[:, np.newaxis]])
        return preds

    def train_model(self, train_config_list, train_feat_data, cell_feat_list):
        step_two_train_x = None
        step_two_train_y = []

        for fold_i, train_config in enumerate(train_config_list):
            # 先做第一阶段训练
            log_msg = f'\n\ntraing fold {fold_i + 1}'
            print(log_msg)
            logger.warning(log_msg)
            if self.debug_mod:
                train_ins = load_json(train_config['train_path'])[:200]
                # train_ins = train_ins
                dev_ins = load_json(train_config['dev_path'])[:200]
            else:
                train_ins = load_json(train_config['train_path'])
                dev_ins = load_json(train_config['dev_path'])
            whole_x = []
            whole_y = []
            for _, unass_pid, pos_aid, neg_aids in train_ins:
                for neg_aid in neg_aids:
                    feat = train_feat_data.get_whole_feat(unass_pid, neg_aid, cell_feat_list)
                    whole_x.append(feat)
                    whole_y.append(0)
                feat = train_feat_data.get_whole_feat(unass_pid, pos_aid, cell_feat_list)
                whole_x.append(feat)
                whole_y.append(1)
            whole_x = np.array(whole_x)
            whole_y = np.array(whole_y)
            self.fit(whole_x, whole_y, 'lv1', fold_i)
            if self.has_lv2:
                # 产生第二阶段训练数据
                tmp_dev_ins = copy.deepcopy(dev_ins)
                new_dev_ins = random_select_instance(tmp_dev_ins, 0.2)
                for _, unass_pid, pos_aid, neg_aids in new_dev_ins:
                    step_one_feat = []
                    for neg_aid in neg_aids:
                        feat = train_feat_data.get_whole_feat(unass_pid, neg_aid, cell_feat_list)
                        step_one_feat.append(feat)
                        step_two_train_y.append(0)
                    if pos_aid != '':
                        feat = train_feat_data.get_whole_feat(unass_pid, pos_aid, cell_feat_list)
                        step_one_feat.append(feat)
                        step_two_train_y.append(1)
                    step_two_feat = self.get_lv2_feat(step_one_feat, fold_i)
                    if step_two_train_x is None:
                        step_two_train_x = step_two_feat
                    else:
                        step_two_train_x = np.vstack([step_two_train_x, step_two_feat])
                del tmp_dev_ins

        if self.has_lv2:
            step_two_train_x = np.array(step_two_train_x)
            step_two_train_y = np.array(step_two_train_y)
            assert len(step_two_train_x) == len(step_two_train_y)
            self.fit(step_two_train_x, step_two_train_y, training_type='lv2')

    def get_lv2_feat(self, candis_feature, fold_i):

        assert self.has_lv2, 'FOR NIL'
        candi_num = len(candis_feature)
        assert candi_num > 0
        lv1_preds_all = self._get_lv1_preds(candis_feature, fold_i)  # [candi_num, lv_1_model_num]
        score_feat_all = None  # [4 * lv_1_model_num]
        for i in range(len(self.lv1_gdb_type)):
            lv1_preds = lv1_preds_all[:, i]
            max_score = np.max(lv1_preds)
            if candi_num > 1:
                min_score = np.min(lv1_preds)
                mean_score = np.mean(lv1_preds)
                lv1_preds[np.argmax(lv1_preds)] = np.min(lv1_preds)
                second_score = np.max(lv1_preds)
                score_feat = [max_score, mean_score,
                              round((max_score - second_score) / (1e-8 + max_score - mean_score), 5),
                              round((max_score - second_score) / (1e-8 + max_score - min_score), 5)]
            else:
                score_feat = [max_score, max_score, 0, 0]
            score_feat = np.array(score_feat).reshape(1, 4).repeat(candi_num, axis=0)
            if score_feat_all is None:
                score_feat_all = score_feat
            else:
                score_feat_all = np.hstack([score_feat_all, score_feat])

        candis_feature = np.hstack([candis_feature, score_feat_all])
        return candis_feature

    def predict(self, candis_feature):
        # candis_feature = np.array(candis_feature)
        if self.has_lv2:
            lv2_feat_all = []
            for fold_i in range(self.kfold):
                lv2_feat = self.get_lv2_feat(candis_feature, fold_i)  # [candi_num,feat_dim ]
                # lv2_feat = lv2_feat[:, np.newaxis, :]  # [candi_num,1,feat_dim ]
                lv2_feat_all.append(lv2_feat)
            lv2_feat = np.mean(lv2_feat_all, axis=0)
            preds = None
            for lv2_model, gdb_type in zip(self.lv2_model_list, self.lv2_gdb_type):
                pred = get_gbd_pred(lv2_model, lv2_feat, gdb_type)[:, np.newaxis]
                if preds is None:
                    preds = pred
                else:
                    preds = np.hstack([preds, pred])
            preds = np.mean(preds, axis=1)
            return preds

        preds_all = None
        for fold_i in range(self.kfold):
            preds = self._get_lv1_preds(candis_feature, fold_i)  # [candi_num, step_one_model_num]
            preds = np.mean(preds, axis=1)[:, np.newaxis]  # [candi_num]
            if preds_all is None:
                preds_all = preds
            else:
                preds_all = np.hstack([preds_all, preds])
        preds_all = np.mean(preds_all, axis=1)
        return preds_all



class GBDTModel:
    def __init__(self,
                 train_config_list,
                 model_save_dir,
                 graph_data=False,
                 debug=False):
        self.train_config_list = train_config_list # k fold train data
        self.model_save_dir = model_save_dir
        #Model configuration
        self.cell_list_config = [
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': ['bert'],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': ['bert'],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': ['bert'],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': [],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': [],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': [],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : ['bert'],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : ['bert'],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : ['bert'],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : [],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : [],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : [],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ]
            ],
        },
    ]
        if graph_data : #Only train one catboost
            self.cell_list_config = []
            self.cell_list_config.append(
                {
                    'cell_weight': 5,
                    'score': 0.0,
                    'feature_list': ['graph'],
                    'vote_type': 'mean',
                    'model': [
                        [
                            {
                                'gbd_type': 'cat',
                                'params': {'verbose': False}
                            }
                        ],
                        [
                        ]
                    ],
                }
            )
        self.cell_weight_sum = 0
        for cell_config in self.cell_list_config:
            self.cell_weight_sum += cell_config['cell_weight']
        self.debug = debug

    def train_cell_model_as_stacking(self,cell_config, train_config_list, train_feat_data: FeatDataLoader,
                                     cell_save_root, cell_index: int):
        '''train one cell '''
        cell_feat_list = cell_config['feature_list']
        cell_model = CellModel(cell_config['model'], len(train_config_list), debug_mod= self.debug)
        cell_model.train_model(train_config_list, train_feat_data, cell_feat_list)
        save_pickle(cell_model, cell_save_root, f'cell-{cell_index}.pkl')
        return cell_model

    def fit(self, train_feature_config):
        ''' train GBDT model'''
        os.makedirs(self.model_save_dir, exist_ok=True)

        train_feat_data = FeatDataLoader(train_feature_config) #train set feature
        cell_model_list = []
        for cell_i, cell_config in enumerate(self.cell_list_config):
            # 先训练每个cell
            s_time = time.time()
            log_msg = f'\n\nbegin to train cell {cell_i + 1}.'
            print(log_msg)
            logger.warning(log_msg)
            train_feat_data.update_feat(cell_config['feature_list'])

            if 'train_config_list' in cell_config:
                in_train_config_list = cell_config['train_config_list']
            else:
                in_train_config_list = self.train_config_list

            cell_model = self.train_cell_model_as_stacking(cell_config, in_train_config_list, train_feat_data,
                                                      self.model_save_dir,
                                                      cell_i + 1)
            cell_model_list.append(cell_model)

        return cell_model_list


    def load(self,cell_model_path_list):
        '''
            load cell models from path
        '''
        cell_model_list = []
        for cell_i, cell_config in enumerate(self.cell_list_config):
            cell_model = pickle.load(open(cell_model_path_list[cell_i], 'rb'))
            cell_model_list.append(cell_model)

        return cell_model_list


