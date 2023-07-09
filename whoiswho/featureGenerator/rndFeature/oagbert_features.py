import sys
from time import time, strftime
from tqdm import tqdm
import json
import random
import pickle
import torch
from cogdl.oag import oagbert
from whoiswho.featureGenerator.rndFeature.model import bertEmbeddingLayer, matchingModel
import argparse
import os
sys.path.append('../../../')
from whoiswho.config import RNDFilePathConfig, configs, version2path,pretrained_oagbert_path
from whoiswho.dataset import load_utils
from whoiswho import logger
debug_mod = False

'''
This file is used to generate bert_simi_feature:
(the similarity between the papers that the candidate author has posted and the paper to be assigned)
Generating bert_simi_feature consists of two steps:
1, Calculating the representation of the papers published by the candidate authors.
2, Calculating the bert_simi_feature.
'''

class ProcessFeature:
    def __init__(self, nameAidPid_path, prosInfo_path, unassCandi_path, validUnassPub_path):

        with open(nameAidPid_path, 'r') as files:
            self.nameAidPid = json.load(files)

        with open(prosInfo_path, 'r') as files:
            self.prosInfo = json.load(files)

        with open(unassCandi_path, 'r') as files:
            self.unassCandi = json.load(files)

        with open(validUnassPub_path, 'r') as files:
            self.validUnassPub = json.load(files)

        if debug_mod:
            self.unassCandi = self.unassCandi[:5]

        self.maxPapers = 40
        global bert_device
        bert_device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.matching_model = matchingModel(bert_device)
        self.matching_model.to(bert_device)
        self.matching_model.eval()


        _, self.bertModel = oagbert( pretrained_oagbert_path)
        self.embedding_model = bertEmbeddingLayer(self.bertModel)
        self.embedding_model.to(bert_device)
        self.embedding_model.eval()

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
            if (ins_author_index == author_index):
                try:
                    orgnizations = ins_author["org"].strip().lower()
                except:
                    orgnizations = ""

                if (orgnizations.strip().lower() != ""):
                    org_str = orgnizations
            else:
                try:
                    name = ins_author["name"].strip().lower()
                except:
                    name = ""
                if (name != ""):
                    name_info.add(name)
        keywords_str = " ".join(keywords_info).strip()
        return (name_info, org_str, venue, keywords_str, title), abstract

    def get_candi_auth_paper_bert_emb(self, start_name_index, end_name_index, candi_4name_bert_emb_path):
        '''
        Function:
            Calculate the representation of the papers published by the candidate authors
            from the start_name_index author name to the end_name_index author name.
        Args:
            start_name_index: the start index of the author name for calculating.
            end_name_index: the end index of the author name for calculating.
            candi_4name_bert_emb_path: the path where saving papers embedding file according to author’s name.
        Returns:
            Saving the representations of these papers as pickle files according to the author's name:
            eg. The format of the file name: "Hong_Li.pickle" is [(aid,[(pid, emb)])].
            When calculating bert_simi_feature,
            these pickles can be reused to avoid repeated calculation.
        '''
        print("Start generating bert embedding...")
        allCandihName = set()
        allCandiNameAidPid = dict()  # {"candiName": [(Aid, [Pid])]}
        if not os.path.exists(candi_4name_bert_emb_path):
            os.makedirs(candi_4name_bert_emb_path, exist_ok=True)

        candi_4name_bert_emb_index_path = f"{candi_4name_bert_emb_path}candi_4name_bert_emb_index.pickle"
        if not os.path.exists(candi_4name_bert_emb_index_path): # if not found, create a new one
            for insIndex in range(len(self.unassCandi)):
                # Calculate the names of all authors involved in the test set
                unassPid, candiName = self.unassCandi[insIndex]
                allCandihName.add(candiName)
                tmpCandiAuthidPidList = []
                candiAuthors = list(self.nameAidPid[candiName].keys())
                for each in candiAuthors:
                    # print("current process: ", insIndex, each)
                    totalPubs = self.nameAidPid[candiName][each]
                    if len(totalPubs) == 0:
                        print("Following author has never published a paper", candiName, each)
                        raise RuntimeError
                    tmp_paper_num = min(len(totalPubs), self.maxPapers)
                    tmpCandiAidPidList = totalPubs[0:tmp_paper_num]
                    tmpCandiAuthidPidList.append((each, tmpCandiAidPidList))
                allCandiNameAidPid[candiName] = tmpCandiAuthidPidList

            with open(candi_4name_bert_emb_index_path, mode="wb") as f_w:
                pickle.dump(allCandiNameAidPid, f_w)
                f_w.close()
        else:
            with open(candi_4name_bert_emb_index_path, mode="rb") as f_r:
                allCandiNameAidPid = pickle.load(f_r)
        # Calculate the paper embedding of each candidate author according to the name and save it as pickle
        # {"candiName": [(Aid, [Pid])]}
        if start_name_index == -1 and end_name_index == -1:
            start_name_index = 0
            end_name_index = len(allCandiNameAidPid)

        current_name_index = 0
        for candiName, aidPidList in allCandiNameAidPid.items():
            if current_name_index < start_name_index:
                current_name_index = current_name_index + 1
                continue
            if current_name_index >= end_name_index:
                break
            if current_name_index >= start_name_index and current_name_index < end_name_index:
                # Current candidate name
                current_name = candiName
                # The format of the last stored representation specific to author name [(aid,[(pid, emb)]]
                tmp_4name_emb_list = []

                for aidPidListItem in aidPidList:
                    current_index = 0

                    current_aid = aidPidListItem[0]
                    pidList = aidPidListItem[1]
                    tmp_4aid_pid_list = []
                    tmp_4aid_emb_list = []

                    for pidItem in pidList:
                        current_pid = pidItem
                        candiPAttrItem, candiPAbstractItem = self.get_paper_attr(pidItem, self.prosInfo)
                        candiP4BertInputItem = self.get_paper_bert_input(candiPAttrItem, candiPAbstractItem)

                        tmp_4aid_pid_list.append(current_pid)
                        current_index = current_index + 1

                        with torch.no_grad():
                            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = candiP4BertInputItem
                            _, unassPBertEmb = self.embedding_model(
                                torch.LongTensor(input_ids).unsqueeze(0).to(bert_device),
                                torch.LongTensor(token_type_ids).unsqueeze(0).to(bert_device),
                                torch.LongTensor(input_masks).unsqueeze(0).to(bert_device),
                                torch.LongTensor(position_ids).unsqueeze(0).to(bert_device),
                                torch.LongTensor(position_ids_second).unsqueeze(0).to(bert_device)
                            )
                            unassPBertEmb_cpu = unassPBertEmb.cpu()
                            tmp_4aid_emb_list.append(unassPBertEmb_cpu)

                    assert len(tmp_4aid_emb_list) == len(tmp_4aid_pid_list)
                    temp_4aid_pid_emb_pair_list = []
                    for i in range(len(tmp_4aid_emb_list)):
                        tmp_pid = tmp_4aid_pid_list[i]
                        tmp_emb = tmp_4aid_emb_list[i]
                        temp_4aid_pid_emb_pair_list.append((tmp_pid, tmp_emb))

                    tmp_4name_emb_list.append((current_aid, temp_4aid_pid_emb_pair_list))
                candi_4name_bert_emb_pickle_path = f"{candi_4name_bert_emb_path}{current_name}_bert_emb.pickle"

                with open(candi_4name_bert_emb_pickle_path, mode="wb") as f_w:
                    # [(aid,[(pid, emb)])]
                    pickle.dump(tmp_4name_emb_list, f_w)
                print("The paper representations of the following author names have been saved: ", current_name)
                current_name_index = current_name_index + 1

    def get_bert_simi_feature(self, start, end, candi_4name_bert_emb_path, bert_simi_save_path,
                              bert_simi_final_save_path):
        '''
        Function:
            Based on OAGBert's representation of the paper and the kernel function to construct the similarity
            feature between the papers published by the candidate authors and the papers to be assigned.
        Args:
            start: the start index of the unassigned paper list.
            end: the end index of the unassigned paper list.
            candi_4name_bert_emb_path: the path where saving papers embedding file according to author’s name.
            bert_simi_save_path: the path where saving temp bert_simi_feature file.
            bert_simi_final_save_path:the path where saving final bert_simi_feature file.
        Returns:
            Saving the bert_simi_feature file as the following format:
            eg. {pid: {aid: bert_simi_feature}}
        '''
        # Calculate the total length of the list of unassigned papers

        print("The total length of the list of unassigned papers: ", len(self.unassCandi))
        if start == -1 and end == -1:
            start = 0
            end = len(self.unassCandi)
        allUnassPCandiAuthPWholeSimi = {}
        # A total of several CandiNames are involved in the calculation, and bert_embedding is obtained
        CandiNameEmb = {}
        # print(len(self.unassCandi))
        for insIndex in range(start, end):
            unassPid, candiName = self.unassCandi[insIndex]
            if candiName not in CandiNameEmb.keys():
                tmp_path = f"{candi_4name_bert_emb_path}{candiName}_bert_emb.pickle"
                if not os.path.exists(tmp_path):
                    print("Please Generating Paper Embedding First!")
                    raise RuntimeError
                    exit(-1)
                with open(tmp_path, mode="rb") as f_r:
                    candiNameBertEmb = pickle.load(f_r)
                    CandiNameEmb[candiName] = candiNameBertEmb
                print("load candiName Emb: ", candiName)
        for insIndex in tqdm(range(start, end)):
            unassPid, candiName = self.unassCandi[insIndex]
            unassPAttr, unassPAbstract = self.get_paper_attr(unassPid, self.validUnassPub)
            unassP4BertInput = self.get_paper_bert_input(unassPAttr, unassPAbstract)

            candiAuthors = list(self.nameAidPid[candiName].keys())

            tmpCandiAuthor = []

            tmpCandiAuthPSimi = {}
            for each in candiAuthors:
                # print("current process: ", insIndex, each)
                # [(aid,[(pid, emb)])]
                candiPBertEmbList = self.get_paper_emb_by_name_aid(candiName, each, CandiNameEmb)

                if len(candiPBertEmbList) == 0:
                    print("Author no paper: ", candiName, each)
                    raise RuntimeError

                tmpCandiAuthor.append(each)

                with torch.no_grad():
                    input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = unassP4BertInput
                    _, unassPBertEmb = self.embedding_model(
                        torch.LongTensor(input_ids).unsqueeze(0).to(bert_device),
                        torch.LongTensor(token_type_ids).unsqueeze(0).to(bert_device),
                        torch.LongTensor(input_masks).unsqueeze(0).to(bert_device),
                        torch.LongTensor(position_ids).unsqueeze(0).to(bert_device),
                        torch.LongTensor(position_ids_second).unsqueeze(0).to(bert_device)
                    )
                    candiPBertEmbs = torch.cat(candiPBertEmbList)
                    whole_sim = self.matching_model(unassPBertEmb, candiPBertEmbs.to(bert_device))
                    tmpCandiAuthPSimi[each] = whole_sim.cpu().numpy()

            allUnassPCandiAuthPWholeSimi[unassPid] = tmpCandiAuthPSimi

        # Save bert_simi feature file
        if not os.path.exists(bert_simi_save_path):
            os.makedirs(bert_simi_save_path, exist_ok=True)

        # get all bert_simi feature at one time
        if start == 0 and end == len(self.unassCandi):
            with open(bert_simi_final_save_path, 'wb') as files:
                pickle.dump(allUnassPCandiAuthPWholeSimi, files)
        # get all bert_simi feature by different start end index
        else:
            with open(f'{bert_simi_save_path}bert_simi_{start}_{end}.pkl', 'wb') as files:
                pickle.dump(allUnassPCandiAuthPWholeSimi, files)
        return allUnassPCandiAuthPWholeSimi

    def get_paper_emb_by_name_aid(self, name, aid, target_dict):
        # {name: [(aid,[(pid, emb)])]}
        target_list = target_dict[name]
        for aid_pid_emb_item in target_list:
            if aid_pid_emb_item[0] == aid:
                target_aid_pid_emb_list = aid_pid_emb_item[1]
                all_aid_embs = [item[1] for item in target_aid_pid_emb_list]
                return all_aid_embs

    def get_paper_bert_input(self, instance_feature, abstract):
        name_info, org_str, venue, keywords_str, title = instance_feature
        # input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = bert_token
        bert_token = self.bertModel.build_inputs(title=title, abstract=abstract, venue=venue, authors=name_info,
                                            concepts=keywords_str, affiliations=org_str)
        return bert_token


def merge_final_simi_pickle(bert_simi_save_path, start_end_index_pair_list, bert_simi_final_save_path):
    '''
    Function:
        Merging the different part of bert_simi feature pickle files.
    Args:
        bert_simi_save_path: The path where saving different part of bert_simi feature pickle files
        start_end_index_pair_list: The list of start index and end index pair
        corresponding to the different part of bert_simi feature pickle files
        eg. start_end_index_pair_list = [(0,3500),(3500,7000),(7000,10500),(10500,13765)]
    Returns:

    '''
    final_simi_dict = {}
    # start_end = [(0,3500),(3500,7000),(7000,10500),(10500,13765)]
    start_end = start_end_index_pair_list
    global_begin = start_end_index_pair_list[0][0]
    global_end = start_end_index_pair_list[-1][-1]
    for start, end in start_end:
        with open(f'{bert_simi_save_path}bert_simi_{start}_{end}.pkl', mode="rb") as f_r:
            tmp_simi_dict = pickle.load(f_r)
            final_simi_dict.update(tmp_simi_dict)
    # bert_simi_save_dictory = bert_simi_save_path[0:-4]
    # bert_simi_save_dictory = bert_simi_save_path
    # with open(f'{bert_simi_save_dictory}bert_simi_{global_begin}_{global_end}.pkl', mode="wb") as f_w:
    #     pickle.dump(final_simi_dict, f_w)
    with open(bert_simi_final_save_path, mode="wb") as f_w:
        pickle.dump(final_simi_dict, f_w)
    print("The merging of the simi feature pickle is complete")


class OagbertFeatures:
    def __init__(self, version, raw_data_root = None, processed_data_root = None,bert_feat_root = None):
        '''
        1.根据type 配置config
        2.根据config 配置processFeature
        '''
        self.v2path = version2path(version)
        self.raw_data_root = raw_data_root
        self.processed_data_root = processed_data_root
        self.bert_feat_root = bert_feat_root

        self.name = self.v2path['name']
        self.task = self.v2path['task']  # RND SND
        assert self.task == 'RND', 'Only support RND task'

        self.type = self.v2path['type']  # train valid test

        # Modifying arguments when calling from outside
        if not raw_data_root:
            self.raw_data_root =  self.v2path['raw_data_root']
        if not processed_data_root:
            self.processed_data_root = self.v2path['processed_data_root']
        if not bert_feat_root:
            self.bert_feat_root = self.v2path['bert_feat_root']

        if self.type == 'train':
            self.config = {
                "nameAidPid_path": self.processed_data_root + 'train/offline_profile.json',
                "prosInfo_path": self.raw_data_root + RNDFilePathConfig.train_pubs,
                "unassCandi_path": self.processed_data_root + RNDFilePathConfig.unass_candi_offline_path,
                "validUnassPub_path": self.raw_data_root + RNDFilePathConfig.train_pubs,
                "candi_4name_bert_emb_path": self.bert_feat_root + 'train/',
                "bert_simi_save_path": self.bert_feat_root + 'train/',
                "bert_simi_final_save_path": self.bert_feat_root + 'pid2aid2bert_feat.offline.pkl',
                "start_end_index_pair_list": [(0, 4000), (4000, 8000), (8000, 12000), (12000, 15873)]
            }
        elif self.type == 'valid':
            self.config = {
                "nameAidPid_path": self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                "prosInfo_path": self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                "unassCandi_path": self.processed_data_root + RNDFilePathConfig.unass_candi_v1_path,
                "validUnassPub_path": self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v1_path,
                "candi_4name_bert_emb_path": self.bert_feat_root + 'online_testv1/',
                "bert_simi_save_path": self.bert_feat_root + 'online_testv1/',
                "bert_simi_final_save_path": self.bert_feat_root + 'pid2aid2bert_feat.onlinev1.pkl',
                "start_end_index_pair_list": [(0, 3463), (3463, 6926), (6926, 10389), (10389, 13849)]
            }
        else:
            self.config = {
                "nameAidPid_path": self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                "prosInfo_path": self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                "unassCandi_path": self.processed_data_root + RNDFilePathConfig.unass_candi_v2_path,
                "validUnassPub_path": self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v2_path,
                "candi_4name_bert_emb_path": self.bert_feat_root + 'online_testv2/',
                "bert_simi_save_path": self.bert_feat_root + 'online_testv2/',
                "bert_simi_final_save_path": self.bert_feat_root + 'pid2aid2bert_feat.onlinev2.pkl',
                "start_end_index_pair_list": [(0, 3700), (3700, 7400), (7400, 11100), (11100, 14560)]
            }
        #Different configs correspond to different ProcessFeatures
        self.genBertSimiFeat = ProcessFeature(self.config["nameAidPid_path"], self.config["prosInfo_path"],
                                              self.config["unassCandi_path"], self.config["validUnassPub_path"])




    def get_oagbert_feature(self):
        '''
        3.processFeature 调用get_candi_auth_paper_bert_emb和get_bert_simi_feature
        4.merge_final_simi_pickle
        '''
        genBertSimiFeat = self.genBertSimiFeat

        start,end = -1,-1
        genBertSimiFeat.get_candi_auth_paper_bert_emb(start, end, self.config["candi_4name_bert_emb_path"])

        genBertSimiFeat.get_bert_simi_feature(start, end, self.config["candi_4name_bert_emb_path"],
                                                      self.config["bert_simi_save_path"],
                                                      self.config["bert_simi_final_save_path"])

        # merge_final_simi_pickle(self.config["bert_simi_save_path"], self.config["start_end_index_pair_list"],
        #                         self.config["bert_simi_final_save_path"])




if __name__ == "__main__":
    data, version = load_utils.LoadData(name="v3", type="train", task='RND')
    oagbert_features = OagbertFeatures(version)
    oagbert_features.get_oagbert_feature()
    logger.info("Finish Train data")

    data, version = load_utils.LoadData(name="v3", type="valid", task='RND')
    oagbert_features = OagbertFeatures(version)
    oagbert_features.get_oagbert_feature()
    logger.info("Finish Valid data")

    data, version = load_utils.LoadData(name="v3", type="test", task='RND')
    oagbert_features = OagbertFeatures(version)
    oagbert_features.get_oagbert_feature()
    logger.info("Finish Test data")
