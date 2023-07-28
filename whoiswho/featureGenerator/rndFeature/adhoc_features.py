import os
import pickle
import random
import sys
import time
from tqdm import tqdm
from collections import defaultdict
import numpy as np
sys.path.append('../../../')

from whoiswho.config import RNDFilePathConfig, configs, version2path
from whoiswho.character.feature_process import featureGeneration
from whoiswho.utils import load_json, save_json, load_pickle, save_pickle
from whoiswho.dataset import load_utils
from whoiswho import logger

# debug_mod = True if sys.gettrace() else False
debug_mod = False


class ProcessFeature:
    def __init__(self, name2aid2pid_path, whole_pub_info_path, unass_candi_path, unass_pubs_path):
        '''

        Args:
            name2aid2pid_path:
            whole_pub_info_path:
            unass_candi_path:
            unass_pubs_path:
        '''
        self.nameAidPid = load_json(name2aid2pid_path)
        self.prosInfo = load_json(whole_pub_info_path)
        self.unassCandi = load_json(unass_candi_path)
        if debug_mod:
            self.unassCandi = self.unassCandi[:5]
        self.validUnassPub = load_json(unass_pubs_path)
        # self.maxNames = 64
        self.maxPapers = 256

    def get_paper_attr(self, pids, pubDict):
        split_info = pids.split('-')
        pid = str(split_info[0])
        author_index = int(split_info[1])
        papers_attr = pubDict[pid]
        name_info = set()
        org_str = ""
        keywords_info = set()
        try:
            title = papers_attr["title"].strip().lower()
        except:
            title = ""

        try:
            venue = papers_attr["venue"].strip().lower()
        except:
            venue = ""
        try:
            abstract = papers_attr["abstract"]
        except:
            abstract = ""

        try:
            keywords = papers_attr["keywords"]
        except:
            keywords = []

        for ins in keywords:
            keywords_info.add(ins.strip().lower())

        paper_authors = papers_attr["authors"]
        for ins_author_index in range(len(paper_authors)):
            ins_author = paper_authors[ins_author_index]
            if ins_author_index == author_index:
                try:
                    orgnizations = ins_author["org"].strip().lower()
                except:
                    orgnizations = ""

                if orgnizations.strip().lower() != "":
                    org_str = orgnizations
            else:
                try:
                    name = ins_author["name"].strip().lower()
                except:
                    name = ""
                if name != "":
                    name_info.add(name)
        keywords_str = " ".join(keywords_info).strip()
        return name_info, org_str, venue, keywords_str, title

    def getUnassFeat(self):
        tmp = []
        tmpCandi = []
        for insIndex in tqdm(range(len(self.unassCandi)),desc='Extracting paper information...'):
            # if insIndex > 30:
            #     break
            unassPid, candiName = self.unassCandi[insIndex]
            unassAttr = self.get_paper_attr(unassPid, self.validUnassPub)
            candiAuthors = list(self.nameAidPid[candiName].keys())

            tmpCandiAuthor = []
            tmpFeat = []
            for each in candiAuthors:
                totalPubs = self.nameAidPid[candiName][each]
                samplePubs = random.sample(totalPubs, min(len(totalPubs), self.maxPapers))
                candiAttrList = [(self.get_paper_attr(insPub, self.prosInfo)) for insPub in samplePubs]
                tmpFeat.append((unassAttr, candiAttrList))
                tmpCandiAuthor.append(each)

            tmp.append((insIndex, tmpFeat))
            tmpCandi.append((insIndex, unassPid, tmpCandiAuthor))
        return tmp, tmpCandi



class AdhocFeatures:
    def __init__(self,version, raw_data_root = None, processed_data_root = None,hand_feat_root = None,graph_data=False):
        self.v2path = version2path(version)
        self.raw_data_root = raw_data_root
        self.processed_data_root = processed_data_root
        self.hand_feat_root = hand_feat_root

        self.name = self.v2path['name']
        self.task = self.v2path['task'] #RND SND
        assert self.task == 'RND' , 'Only support RND task'

        self.type = self.v2path['type'] #train valid test

        #Modifying arguments when calling from outside
        if not raw_data_root:
            self.raw_data_root = self.v2path['raw_data_root']
        if not processed_data_root:
            self.processed_data_root = self.v2path['processed_data_root']
        if not hand_feat_root:
            self.hand_feat_root = self.v2path['hand_feat_root']


        if self.type == 'train':
            self.config = {
                'name2aid2pid_path': self.processed_data_root + 'train/offline_profile.json',
                'whole_pub_info_path': self.raw_data_root + RNDFilePathConfig.train_pubs,
                'unass_candi_path': self.processed_data_root + RNDFilePathConfig.unass_candi_offline_path,
                'unass_pubs_path': self.raw_data_root + RNDFilePathConfig.train_pubs,
            }
            self.feat_save_path = self.hand_feat_root + 'pid2aid2hand_feat.offline.pkl'

        elif self.type == 'valid':
            self.config = {
                'name2aid2pid_path': self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                'whole_pub_info_path': self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                'unass_candi_path': self.processed_data_root + RNDFilePathConfig.unass_candi_v1_path,
                'unass_pubs_path': self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v1_path,
            }
            self.feat_save_path = self.hand_feat_root + 'pid2aid2hand_feat.onlinev1.pkl'
        else:
            self.config = {
                'name2aid2pid_path'  : self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                'whole_pub_info_path': self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                'unass_candi_path'   : self.processed_data_root + RNDFilePathConfig.unass_candi_v2_path,
                'unass_pubs_path'    : self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v2_path,
            }

        if graph_data == True:
            self.whoiswhograph_extend_processed_data = self.v2path['whoiswhograph_extend_processed_data']
            self.config = {
                'name2aid2pid_path': self.whoiswhograph_extend_processed_data + 'train/offline_profile_by_wholeprofile.json',
                'whole_pub_info_path': self.raw_data_root + RNDFilePathConfig.train_pubs,
                'unass_candi_path': self.whoiswhograph_extend_processed_data + RNDFilePathConfig.unass_candi_offline_path,
                'unass_pubs_path': self.raw_data_root + RNDFilePathConfig.train_pubs,
            }
            self.feat_save_path = self.hand_feat_root + 'whoiswhograph_pid2aid2hand_feat.offline.pkl'


        self.genAdhocFeat = ProcessFeature(**self.config)


    def get_hand_feature(self):
        s_time = time.time()
        genAdhocFeat= self.genAdhocFeat
        genFeatures = featureGeneration()

        rawFeatData, unassCandiAuthor = genAdhocFeat.getUnassFeat()
        print('begin multi_process_data')
        hand_feature_list = genFeatures.multi_process_data(rawFeatData)
        print('end multi_process_data')
        assert len(hand_feature_list) == len(unassCandiAuthor)
        pid2aid2cb_feat = defaultdict(dict)
        for hand_feat_item, candi_item in zip(hand_feature_list, unassCandiAuthor):
            ins_index, unass_pid, candi_aids = candi_item
            hand_feat_list, coauthor_ratio = hand_feat_item
            assert len(hand_feat_list) == len(candi_aids)
            for candi_aid, hand_f in zip(candi_aids, hand_feat_list):
                pid2aid2cb_feat[unass_pid][candi_aid] = np.array(hand_f)
        pid2aid2cb_feat = dict(pid2aid2cb_feat)
        print("process data: %.6f" % (time.time() - s_time))
        save_pickle(pid2aid2cb_feat, self.feat_save_path)

if __name__ == '__main__':
    version = {"name": 'v3', "task": 'RND', "type": 'train'}
    adhoc_features = AdhocFeatures(version,graph_data=True)
    adhoc_features.get_hand_feature()
    logger.info("Finish Train data")

    version = {"name": 'v3', "task": 'RND', "type": 'valid'}
    adhoc_features = AdhocFeatures(version)
    adhoc_features.get_hand_feature()
    logger.info("Finish Valid data")

    version = {"name": 'v3', "task": 'RND', "type": 'test'}
    adhoc_features = AdhocFeatures(version)
    adhoc_features.get_hand_feature()
    logger.info("Finish Test data")

