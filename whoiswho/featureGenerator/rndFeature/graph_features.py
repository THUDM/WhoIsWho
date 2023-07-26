import pickle
import sys
sys.path.append('../../../')
from whoiswho.utils import load_json, save_json, load_pickle, save_pickle
import random
import json
import os
from os.path import join,dirname
import copy
import numpy as np
from functools import reduce
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import argparse
from torch_geometric.data.batch import Batch
from whoiswho.featureGenerator.rndFeature.graph_dataloader import GraphPairDataset
from whoiswho.featureGenerator.rndFeature.model import SAGE, GAT,infonce,matchingModel,matchingModel_gnn,gat_parameters,sage_parameters,HR,MRR
from whoiswho.config import RNDFilePathConfig, configs, version2path,uuid_path
from whoiswho import logger


class ProcessFeature:
    def __init__(self, nameAidPid_path, prosInfo_path, unassCandi_path, validUnassPub_path,  dataset_type, gpu_device=0):

        with open(nameAidPid_path, 'r') as files:
            self.nameAidPid = json.load(files)

        with open(prosInfo_path, 'r') as files:
            self.prosInfo = json.load(files)

        with open(unassCandi_path, 'r') as files:
            self.unassCandi = json.load(files)

        with open(validUnassPub_path, 'r') as files:
            self.validUnassPub = json.load(files)

        with open(uuid_path, 'r') as files:
            self.aminer_uuid = json.load(files)
            self.uuid_aminer = dict(zip(self.aminer_uuid.values(), self.aminer_uuid.keys()))

        self.dataset_type = dataset_type
        global device
        device = torch.device(f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu')

        self.matching_model = matchingModel(device)
        self.matching_model.to(device)
        self.matching_model.eval()

        model_para = gat_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        #load saved gnn model
        self.gnnmodel = GAT(in_channels=768, out_channels=128, **model_para).to(device)
        model_dict = torch.load(join(os.path.abspath(dirname(__file__)),'gat_model_oagbert_sim.pt'),map_location='cpu')['Gat']
        self.gnnmodel.load_state_dict(model_dict)
        self.gnnmodel.eval()


    def get_candi_auth_graph_path(self, candi_4name_graph_path,train_author_path,valid_author_path):
        '''get the candidate graph path '''
        # {"candiName": [(Aid, author_graph_path)...]}
        allCandiNameAidPid = dict()
        if not os.path.exists(candi_4name_graph_path):
            os.makedirs(candi_4name_graph_path, exist_ok=True)
        candi_4name_graph_index_path = f"{candi_4name_graph_path}candi_4name_graphpath.json"
        #Used to determine if the author's name has graph
        train_author_list = os.listdir(train_author_path)
        train_author_dict = dict(zip(train_author_list,list(range(len(train_author_list)))) )

        valid_author_list =os.listdir(valid_author_path)
        valid_author_dict = dict(zip(valid_author_list,list(range(len(valid_author_list)))) )

        if not os.path.exists(candi_4name_graph_index_path):
            for insIndex in range(len(self.unassCandi)):
                unassPid, candiName = self.unassCandi[insIndex]

                tmpCandiAuthidPidList = []
                candiAuthors = list(self.nameAidPid[candiName].keys())

                for each in candiAuthors:
                    author = self.uuid_aminer[each] + '-' + candiName
                    if author in train_author_dict:
                        graphpath= train_author_path + author

                    elif author in valid_author_dict:
                        graphpath = valid_author_path + author

                    tmpCandiAuthidPidList.append((each, graphpath))
                allCandiNameAidPid[candiName] = tmpCandiAuthidPidList

            save_json(allCandiNameAidPid, candi_4name_graph_index_path)

    def get_graph_simi_feature(self, start, end, candi_4name_graph_path, train_paper_path, valid_paper_path, test_paper_path,
                               graph_simi_save_path, graph_simi_final_save_path, embedding_path, need_more_paper=False):
        # Calculate the total length of the list of unassigned papers
        print("The total length of the list of unassigned papers: ", len(self.unassCandi))
        if start == -1 and end == -1:
            start = 0
            end = len(self.unassCandi)

        if self.dataset_type == "train":
            path = train_paper_path
        elif self.dataset_type == "valid":
            path = valid_paper_path
        else :
            path = test_paper_path

        all_train_emb_path = embedding_path + 'train/all_train_emb_sim.npy'
        logger.info("Loading training set embedding...")
        all_train_emb = np.load(all_train_emb_path, allow_pickle=True).item()

        all_valid_emb_path = embedding_path + 'valid/all_valid_emb_sim.npy'
        all_test_emb_path = embedding_path + 'test/all_test_emb_sim.npy'
        logger.info("Loading valid set embedding...")
        all_valid_emb = np.load(all_valid_emb_path, allow_pickle=True).item()
        logger.info("Loading test set embedding...")
        all_test_emb = np.load(all_test_emb_path, allow_pickle=True).item()
        logger.info("Loading complete！")
        all_emb={}
        all_emb.update(all_train_emb)
        all_emb.update(all_valid_emb)
        all_emb.update(all_test_emb)
        with  open(candi_4name_graph_path+'candi_4name_graphpath.json', 'r', encoding='utf-8') as f:
            CandiNameGraph = json.load(f)

        print(f'whether use more reference papers? {need_more_paper}')

        allUnassPCandiAuthPWholeSimi = {}

        for insIndex in tqdm(range(start, end)):
            pair = []
            unassPid, candiName = self.unassCandi[insIndex]
            unass_path = path + self.uuid_aminer[unassPid.split('-')[0]]
            pair.append(unass_path)
            candiAuthors = list(self.nameAidPid[candiName].keys())
            for each in candiAuthors:
                author_path = self.get_graph_by_name_aid(candiName,each,CandiNameGraph)
                pair.append(author_path)

            graph_data = GraphPairDataset(all_emb=all_emb,need_more_paper=need_more_paper)
            graph_pair,graph_names,paper_emb_len = graph_data.load_graph_pair(pair)
            # print(len(graph_pair))

            length_list = [0] + [len(graph_pair[i].x) for i in range(len(graph_pair))]
            graph_index = [reduce(lambda a, b: a + b, length_list[:i]) for i in range(2, len(length_list) + 1)]
            batch_graph = Batch.from_data_list(graph_pair)

            with torch.no_grad():
                batch_graph.to(device)
                out = self.gnnmodel(batch_graph.x, batch_graph.edge_index)

            tmpCandiAuthPSimi = {}
            if paper_emb_len == 0:
                paper_emb = out[0].unsqueeze(0)  # (1,128)
            else:
                paper_emb = out[:paper_emb_len]
                paper_emb = torch.mean(paper_emb, dim=0, keepdim=True) # (1,128)

            #author_emb & author id
            for idx,each in zip(range(len(graph_index) - 1),candiAuthors):
                author_emb = out[graph_index[idx]:graph_index[idx + 1]]
                whole_sim = self.matching_model(paper_emb.to(device), author_emb.to(device))
                tmpCandiAuthPSimi[each] = whole_sim.cpu().numpy()

            allUnassPCandiAuthPWholeSimi[unassPid] = tmpCandiAuthPSimi

        if not os.path.exists(graph_simi_save_path):
            os.makedirs(graph_simi_save_path, exist_ok=True)

        # get all graph feature at one time
        if start == 0 and end == len(self.unassCandi):
            with open(graph_simi_final_save_path, 'wb') as files:
                pickle.dump(allUnassPCandiAuthPWholeSimi, files)
        # get all graph feature by different start end index
        else:
            with open(f'{graph_simi_save_path}graph_simi_{start}_{end}.pkl', 'wb') as files:
                pickle.dump(allUnassPCandiAuthPWholeSimi, files)

    def get_graph_by_name_aid(self, name, aid, target_dict):
        # target_dict {name: [(aid,author_graph)]}
        target_list = target_dict[name]
        for aid_path_item in target_list:
            if aid_path_item[0] == aid: #Locate the author being queried
                target_author_path = aid_path_item[1]
                return target_author_path



class GraphFeatures:
    def __init__(self, version, raw_data_root = None, processed_data_root = None,
                 graph_data_root = None,whoiswhograph_extend_processed_data = None,graph_feat_root = None,device=0):
        self.v2path = version2path(version)
        self.raw_data_root = raw_data_root
        self.processed_data_root = processed_data_root
        self.graph_data_root = graph_data_root
        self.whoiswhograph_extend_processed_data = whoiswhograph_extend_processed_data
        self.graph_feat_root = graph_feat_root


        self.name = self.v2path['name']
        self.task = self.v2path['task']  # RND SND
        assert self.task == 'RND',  'Only support RND task'

        self.type = self.v2path['type']  # train valid test

        # Modifying arguments when calling from outside
        if not raw_data_root:
            self.raw_data_root =  self.v2path['raw_data_root']
        if not processed_data_root:
            self.processed_data_root = self.v2path['processed_data_root']
        if not graph_data_root:
            self.graph_data_root = self.v2path['whoiswhograph_data_root']
        if not whoiswhograph_extend_processed_data:
            self.whoiswhograph_extend_processed_data = self.v2path['whoiswhograph_extend_processed_data']
        if not graph_feat_root:
            self.graph_feat_root = self.v2path['graph_feat_root']

        self.embedding_path = join(self.v2path['whoiswhograph_emb_root'])

        if self.type == 'train':
            self.config = {
                "nameAidPid_path": self.whoiswhograph_extend_processed_data + 'train/offline_profile.json',
                "prosInfo_path": self.raw_data_root + RNDFilePathConfig.train_pubs,
                "unassCandi_path": self.whoiswhograph_extend_processed_data + RNDFilePathConfig.unass_candi_offline_path,
                "validUnassPub_path": self.raw_data_root + RNDFilePathConfig.train_pubs,

                # graph path
                "prefix_train_author_graph_path": self.graph_data_root + 'train_graph/author_graph/',
                "prefix_train_paper_graph_path": self.graph_data_root +'train_graph/paper_graph/',
                "prefix_valid_author_graph_path": self.graph_data_root +'valid_graph/author_graph/',
                "prefix_valid_paper_graph_path": self.graph_data_root +'valid_graph/paper_graph/',
                "prefix_test_paper_graph_path": self.graph_data_root +'test_graph/paper_graph/',

                "candi_4name_graph_path": self.graph_feat_root + 'train/',
                "graph_simi_save_path": self.graph_feat_root + 'train/',
                "graph_simi_final_save_path": self.graph_feat_root + 'pid2aid2graph_feat_gat.offline.pkl',
                "start_end_index_pair_list": [(0, 2000), (2000, 4000), (4000, 6000), (6000, 7871)]
            }

        elif self.type == "valid":
            self.config = {
                "nameAidPid_path": self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                "prosInfo_path": self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                "unassCandi_path": self.whoiswhograph_extend_processed_data + RNDFilePathConfig.unass_candi_v1_path,
                "validUnassPub_path": self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v1_path,

                # graph path
                "prefix_train_author_graph_path": self.graph_data_root + 'train_graph/author_graph/',
                "prefix_train_paper_graph_path": self.graph_data_root +'train_graph/paper_graph/',
                "prefix_valid_author_graph_path": self.graph_data_root +'valid_graph/author_graph/',
                "prefix_valid_paper_graph_path": self.graph_data_root +'valid_graph/paper_graph/',
                "prefix_test_paper_graph_path": self.graph_data_root +'test_graph/paper_graph/',

                "candi_4name_graph_path": self.graph_feat_root +'online_testv1/',
                "graph_simi_save_path": self.graph_feat_root +'online_testv1/',
                "graph_simi_final_save_path": self.graph_feat_root +'pid2aid2graph_feat_gat.onlinev1.pkl',
                "start_end_index_pair_list": [(0, 3000), (3000, 5939)]
            }

        else:
            self.config = {
                "nameAidPid_path": self.processed_data_root + RNDFilePathConfig.whole_name2aid2pid,
                "prosInfo_path": self.processed_data_root + RNDFilePathConfig.whole_pubsinfo,
                "unassCandi_path": self.whoiswhograph_extend_processed_data + RNDFilePathConfig.unass_candi_v2_path,
                "validUnassPub_path": self.raw_data_root + RNDFilePathConfig.unass_pubs_info_v2_path,

                # graph path
                "prefix_train_author_graph_path": self.graph_data_root + 'train_graph/author_graph/',
                "prefix_train_paper_graph_path": self.graph_data_root +'train_graph/paper_graph/',
                "prefix_valid_author_graph_path": self.graph_data_root +'valid_graph/author_graph/',
                "prefix_valid_paper_graph_path": self.graph_data_root +'valid_graph/paper_graph/',
                "prefix_test_paper_graph_path": self.graph_data_root +'test_graph/paper_graph/',

                "candi_4name_graph_path": self.graph_feat_root +'online_testv2/',
                "graph_simi_save_path": self.graph_feat_root +'online_testv2/',
                "graph_simi_final_save_path": self.graph_feat_root +'pid2aid2graph_feat_gat.onlinev2.pkl',
                "start_end_index_pair_list": [(0, 3000), (3000, 6137)]
            }

        # Different configs correspond to different ProcessFeatures
        self.genGraphSimiFeat = ProcessFeature(self.config["nameAidPid_path"], self.config["prosInfo_path"],
                                              self.config["unassCandi_path"], self.config["validUnassPub_path"],
                                              dataset_type = self.type,gpu_device=device)



    def get_graph_feature(self,start=-1,end=-1):
        genGraphSimiFeat = self.genGraphSimiFeat
        genGraphSimiFeat.get_candi_auth_graph_path(self.config["candi_4name_graph_path"],
                                                   self.config["prefix_train_author_graph_path"],
                                                   self.config["prefix_valid_author_graph_path"])
        genGraphSimiFeat.get_graph_simi_feature(start, end, self.config["candi_4name_graph_path"],
                                                        self.config["prefix_train_paper_graph_path"],
                                                        self.config["prefix_valid_paper_graph_path"],
                                                        self.config["prefix_test_paper_graph_path"],
                                                        self.config["graph_simi_save_path"],
                                                        self.config["graph_simi_final_save_path"],
                                                        embedding_path = self.embedding_path,
                                                        need_more_paper= True)


if __name__ == '__main__':
    version = {"name": 'v3', "task": 'RND', "type": 'train'}
    graph_features = GraphFeatures(version)
    graph_features.get_graph_feature()
    logger.info("Finish Train data")

    version = {"name": 'v3', "task": 'RND', "type": 'valid'}
    graph_features = GraphFeatures(version)
    graph_features.get_graph_feature()
    logger.info("Finish Valid data")

    version = {"name": 'v3', "task": 'RND', "type": 'test'}
    graph_features = GraphFeatures(version)
    graph_features.get_graph_feature()
    logger.info("Finish Test data")