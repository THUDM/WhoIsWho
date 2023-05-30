import numpy as np
from os.path import join
import os
import pickle
import sys
import copy
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from typing import List, Dict, Any
sys.path.append('../../')
from whoiswho import logger
from whoiswho.dataset import load_utils
from whoiswho.dataset.data_process import read_pubs,read_raw_pubs
from whoiswho.config import version2path
from whoiswho.utils import set_log, load_pickle, save_pickle, load_json, save_json
from whoiswho.loadmodel.ClusterModels import DBSCANModel
from whoiswho.featureGenerator.sndFeature.semantic_features import SemanticFeatures
from whoiswho.featureGenerator.sndFeature.relational_features import RelationalFeatures

def tanimoto(p, q):
    """Calculate the tanimoto coefficient.
    Args:
        two texts.
    Returns:
        A float number of coefficient.
    """
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))

def dump_result(pubs, pred):
    """Dump results file.

    Args:
        pubs: papers of this name (List).
        pred: predicted labels (Numpy Array).
    """
    result = []
    for i in set(pred):
        oneauthor = []
        for idx, j in enumerate(pred):
            if i == j:
                oneauthor.append(pubs[idx])
        result.append(oneauthor)
    return result # List[List[pid]]

class SNDTrainer:
    def __init__(self, version, processed_data_root = None, w_author = 1.5,
                 w_org = 1.0, w_venue=1.0, w_title= 0.33, text_weight=1.0,
                 db_eps = 0.2,db_min = 4):
        self.v2path = version2path(version)
        self.name = self.v2path['name']
        self.task = self.v2path['task']  # RND SND
        assert self.task == 'SND', 'This features' \
                                   'only support SND task'
        self.type = self.v2path['type']  # train valid test

        # Modifying arguments when calling from outside
        self.processed_data_root = processed_data_root
        if not processed_data_root:
            self.processed_data_root = self.v2path['processed_data_root']
        self.raw_data_root = self.v2path['raw_data_root']
        self.w_author =  w_author
        self.w_org = w_org
        self.w_venue = w_venue
        self.w_title = w_title
        self.text_weight = text_weight

        self.semantic_feature = SemanticFeatures()
        self.relational_feature = RelationalFeatures(version,self.processed_data_root)

        self.model =  DBSCANModel(db_eps,db_min)


    def save_pair(self,pubs, mode, name, outlier):
        """Save post-matching paper pair by threshold.
        """
        dirpath = join(self.processed_data_root, 'relations', mode, name)

        paper_org = {}
        paper_conf = {}
        paper_author = {}
        paper_word = {}

        temp = set()
        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_org:
                    paper_org[p] = []
                paper_org[p].append(a)
        temp.clear()

        with open(dirpath + "/paper_venue.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_conf:
                    paper_conf[p] = []
                paper_conf[p] = a
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_author:
                    paper_author[p] = []
                paper_author[p].append(a)
        temp.clear()

        with open(dirpath + "/paper_title.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_word:
                    paper_word[p] = []
                paper_word[p].append(a)
        temp.clear()

        paper_paper = np.zeros((len(pubs), len(pubs)))
        for i, pid in enumerate(pubs):
            if i not in outlier:
                continue
            for j, pjd in enumerate(pubs):
                if j == i:
                    continue
                ca = 0
                cv = 0
                co = 0
                ct = 0

                if pid in paper_author and pjd in paper_author:
                    ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * self.w_author
                if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                    cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd])) * self.w_venue
                if pid in paper_org and pjd in paper_org:
                    co = tanimoto(set(paper_org[pid]), set(paper_org[pjd])) * self.w_org
                if pid in paper_word and pjd in paper_word:
                    ct = len(set(paper_word[pid]) & set(paper_word[pjd])) * self.w_title

                paper_paper[i][j] = ca + cv + co + ct

        return paper_paper

    def post_match(self,pred, tcp, cp, pubs, mode, name):
        """Post-match outliers of clustering results.
        Using threshold based characters' matching.

        Args:
            clustering labels (Numpy Array).

        Returns:
            predicted labels (Numpy Array).

        """
        outlier = set()
        for i in range(len(pred)):
            if pred[i] == -1:
                outlier.add(i)
        for i in tcp:
            outlier.add(i)
        for i in cp:
            outlier.add(i)

        paper_pair = self.save_pair(pubs, mode, name, outlier)
        paper_pair1 = paper_pair.copy()
        K = len(set(pred))
        for i in range(len(pred)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])

            while j in outlier:
                paper_pair[i][j] = -1

                last_j = j
                j = np.argmax(paper_pair[i])
                if j == last_j:
                    break

            if paper_pair[i][j] >= 1.5:
                pred[i] = pred[j]
            else:
                pred[i] = K
                K = K + 1

        for ii, i in enumerate(outlier):
            for jj, j in enumerate(outlier):
                if jj <= ii:
                    continue
                else:
                    if paper_pair1[i][j] >= 1.5:
                        pred[j] = pred[i]
        return pred

    def fit(self, add_sem=True, add_rel=True, if_post_match=True,
        add_a=True, add_o=True, add_v=True) :
        pubs = read_pubs(self.raw_data_root,self.type)
        raw_pubs = read_raw_pubs(self.raw_data_root,self.type)
        result = {}

        cur_time = datetime.now().strftime("%m%d%H%M")
        for n, name in enumerate(tqdm(raw_pubs)):
            if self.type == 'train':
                pubs = []
                # ilabel = 0
                # labels = []
                for aid in raw_pubs[name]:
                    pubs.extend(raw_pubs[name][aid])
                    # labels.extend([ilabel] * len(raw_pubs[name][aid]))
                    # ilabel += 1
            elif self.type == 'valid' or 'test':
                # valid or test
                pubs = raw_pubs[name]
            else:
                print("Invalid type!")

            tcp = set()
            cp = set()
            # 逐name获取特征
            if add_sem and not add_rel:
                sem_dis, tcp = self.semantic_feature.cal_semantic_similarity(pubs, name)
                dis = sem_dis
            elif not add_sem and add_rel:
                rel_dis, cp = self.relational_feature.cal_relational_similarity(pubs, name, self.type, add_a, add_o, add_v)
                dis = rel_dis
            elif add_sem and add_rel:
                sem_dis, tcp = self.semantic_feature.cal_semantic_similarity(pubs, name)
                rel_dis, cp = self.relational_feature.cal_relational_similarity(pubs, name, self.type, add_a, add_o, add_v)
                dis = (np.array(rel_dis) + self.text_weight * np.array(sem_dis)) / (1 + self.text_weight)
            # 逐name cluster  结果为pred
            pred = self.model.fit(dis)
            if if_post_match:
                pred = self.post_match(pred, tcp, cp, pubs, self.type, name)


            # 逐name保存聚类结果
            result[name] = []
            result[name].extend(dump_result(pubs, pred))

        save_dir = './whoiswho/training/snd_result'
        os.makedirs(save_dir, exist_ok=True)
        save_json(result,save_dir, f'result.{self.type}.json')


